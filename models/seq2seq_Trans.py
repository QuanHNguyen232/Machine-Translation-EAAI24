# https://www.kaggle.com/code/latinchakma/transformer-based-seq2seq-de-to-en-multi30k
# check hq.nguyen1115@gmail.com for latest ver.
# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

import os
import math
import copy

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer

from .infer_utils import sent2tensor, idx2sent

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network 
class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg, in_lang, out_lang, src_pad_idx, device, verbose=False):
                #  num_encoder_layers: int,
                #  num_decoder_layers: int,
                #  emb_size: int,
                #  nhead: int,
                #  src_vocab_size: int,
                #  tgt_vocab_size: int,
                #  dim_feedforward: int = 512,
                #  dropout: float = 0.1):
        super().__init__()
        
        self.cfg = copy.deepcopy(cfg)
        self.cfg.pop('piv', '')
        self.cfg.pop('tri', '')
        self.cfg['seq2seq']['model_lang'] = self.model_lang = {"in_lang": in_lang, "out_lang": out_lang}
        self.cfg['model_id'] = self.modelname = 'Trans_' + '-'.join(list(self.model_lang.values())) + '_' + cfg['model_id']
        self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])

        self.num_encoder_layers = cfg['seq2seq']['NUM_ENCODER_LAYERS']
        self.num_decoder_layers = cfg['seq2seq']['NUM_DECODER_LAYERS']
        self.emb_size = cfg['seq2seq']['EMB_SIZE']
        self.nhead = cfg['seq2seq']['NHEAD']
        self.src_vocab_size = cfg['seq2seq'][f'{in_lang}_DIM']
        self.tgt_vocab_size = cfg['seq2seq'][f'{out_lang}_DIM']
        self.dim_feedforward = cfg['seq2seq']['FFN_HID_DIM']
        self.dropout = cfg['seq2seq']['TRANS_DROPOUT']
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.verbose = verbose

        self.transformer = Transformer(d_model=self.emb_size,
                                       nhead=self.nhead,
                                       num_encoder_layers=self.num_encoder_layers,
                                       num_decoder_layers=self.num_decoder_layers,
                                       dim_feedforward=self.dim_feedforward,
                                       dropout=self.dropout)
        self.generator = nn.Linear(self.emb_size, self.tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(self.src_vocab_size, self.emb_size)
        self.tgt_tok_emb = TokenEmbedding(self.tgt_vocab_size, self.emb_size)
        self.positional_encoding = PositionalEncoding(self.emb_size, dropout=self.dropout)
        
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, batch, model_cfg=None, criterion=None, teacher_forcing_ratio=None):
        (src, _), (trg, _) = self.prep_input(batch, model_cfg)
        src, trg = src.to(self.device), trg.to(self.device)
        trg_in = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, trg_in)
        memory_key_padding_mask = src_padding_mask
        if self.verbose: print('src', src[:, 0])
        if self.verbose: print('src_mask', src_mask[0])
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_in))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # outs = self.decode(tgt_emb, self.encode(src, src_mask), tgt_mask)
        # outs.shape = [seq_len, batch_size, dim_feedforward]

        logits = self.generator(outs)
        # x = self.encode_train(src, trg)
        # logits = self.decode_train(**x)

        if criterion != None:
            loss = self.compute_loss(logits, trg, criterion)
            return loss, logits, None
        return logits, None
    
    def encode_train(self, src: Tensor, trg: Tensor):
        '''
        src = (S, N)
        trg = (T, N)
        src_emb = (S, N, E)
        tgt_emb = (T, N, E)

        src_mask = (S, S)
        tgt_mask = (T, T)

        src_padding_mask = (N, S)
        tgt_padding_mask = (N, T)

        memory = (S, N, E)
        memory_key_padding_mask = (N, S)
        '''
        src, trg = src.to(self.device), trg.to(self.device)
        trg_in = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, trg_in)
        
        memory_key_padding_mask = src_padding_mask

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_in))

        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        return {
            "tgt_emb": tgt_emb,
            "memory": memory,
            "tgt_mask": tgt_mask,
            "tgt_padding_mask": tgt_padding_mask,
            "memory_key_padding_mask": memory_key_padding_mask
        }
    
    def decode_train(self, tgt_emb, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        logits = self.generator(outs)
        return logits

    def encode(self, src: Tensor, src_mask: Tensor):
        # return: memory.shape = (S, N, E)
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # return out.shape = (T, N, E)
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_loss(self, logits, tgt, criterion):
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        return loss

    def create_mask(self, src, tgt):
        '''
        src = (S, N)
        trg = (T, N)

        src_mask = (S, S)
        tgt_mask = (T, T)

        src_padding_mask = (N, S)
        tgt_padding_mask = (N, T)
        '''
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == PAD_ID).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_ID).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def prep_input(self, batch, model_cfg):
        '''
        batch: dict of langs. each lang is tuple (seq_batch, seq_lens)
        '''
        return (
          batch[model_cfg['seq2seq']['model_lang']['in_lang']],
          batch[model_cfg['seq2seq']['model_lang']['out_lang']]
        )
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        # src.shape = (S, N)
        # src_mask.shape = (S, S)
        src, src_mask = src.to(self.device), src_mask.to(self.device)
        batch_size = src.shape[1]
        memory = self.encode(src, src_mask) # memory.shape = (S, N, E)
        ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device) # memory.shape = (S, N, E)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
            out = self.decode(ys, memory, tgt_mask) # out.shape = (T, N, E)
            out = out.transpose(0, 1) # out.shape = (N, T, E)
            # out[:, -1, :].shape = (N, E) (take last token)
            prob = self.generator(out[:, -1, :]) # shape = (N, out_dim)
            next_word = torch.argmax(prob, dim=1) # shape = (N)
            ys = torch.cat((ys, next_word.unsqueeze(0).type_as(src.data)), dim=0)
            # https://stackoverflow.com/questions/61101919/how-can-i-add-an-element-to-a-pytorch-tensor-along-a-certain-dimension
        return ys # shape = (T, N)

    def translate(self, src_sentences, tkzer_dict, field_dict):
        self.eval()
        tkzer_src = tkzer_dict[self.cfg['seq2seq']['model_lang']['in_lang']]
        field_src = field_dict[self.cfg['seq2seq']['model_lang']['in_lang']]
        field_trg = field_dict[self.cfg['seq2seq']['model_lang']['out_lang']]
        sents = []
        for src_sentence in src_sentences:
            src_, _ = sent2tensor(tkzer_src, field_src, field_trg, self.device, 64, src_sentence)
            sents.append(src_) # src_sent = (S, 1) -> pad = (S, N, 1)
        src = torch.nn.utils.rnn.pad_sequence(sents, padding_value=PAD_ID).squeeze(2) # shape = (S, N)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, num_tokens+5, SOS_ID) # shape = (T, N)
        tgt_tokens, _ = self.process_output(tgt_tokens)
        results = idx2sent(field_trg, tgt_tokens)
        return {'results': results, 'tokens': tgt_tokens}
    
    def process_output(self, output):
        # output = [trg len, batch size, output dim]
        # trg = [trg len, batch size]
        # Process output1 to be input for 
        if output.ndim == 3:
            seq_len, N, _ = output.shape
            tmp_out = output.argmax(2)  # tmp_out = [seq_len, batch_size]
        else:
            seq_len, N = output.shape
            tmp_out = output
        # re-create pivot as src for model2
        piv = torch.zeros_like(tmp_out).type(torch.long).to(output.device)
        piv[0, :] = torch.full_like(piv[0, :], SOS_ID)  # fill all first idx with sos_token

        for i in range(1, seq_len):  # for each i in seq_len
            # if tmp_out's prev is eos_token, replace w/ pad_token, else current value
            eos_mask = (tmp_out[i-1, :] == EOS_ID)
            piv[i, :] = torch.where(eos_mask, PAD_ID, tmp_out[i, :])
            # if piv's prev is pad_token, replace w/ pad_token, else current value
            pad_mask = (piv[i-1, :] == PAD_ID)
            piv[i, :] = torch.where(pad_mask, PAD_ID, piv[i, :])

        # Trim down extra pad tokens
        tensor_list = [piv[i] for i in range(seq_len) if not all(piv[i] == PAD_ID)]  # tensor_list = [new_seq_len, batch_size]
        piv = torch.stack([x for x in tensor_list], dim=0).type(torch.long).to(output.device)
        assert not all(piv[-1] == PAD_ID), 'Not completely trim down tensor'

        # get seq_id + eos_tok id of each sequence
        piv_ids, eos_ids = (piv.permute(1, 0) == EOS_ID).nonzero(as_tuple=True)  # piv_len = [N]
        piv_len = torch.full_like(piv[0], seq_len).type(torch.long)  # init w/ longest seq
        piv_len[piv_ids] = eos_ids + 1 # seq_len = eos_tok + 1

        return piv, piv_len
    
    
    # def greedy_decode(self, src, src_mask, max_len, start_symbol):
    #     # src.shape = (S, 1) as (S, N)
    #     # src_mask.shape = (S, S)
    #     src, src_mask = src.to(self.device), src_mask.to(self.device)
    #     memory = self.encode(src, src_mask) # memory.shape = (S, 1, E) as (S, N, E)
    #     if self.verbose: print('greedy_decode:', 'src.shape', src.shape, 'src_mask.shape', src_mask.shape, 'memory.shape', memory.shape)
    #     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device) # ys.shape = (1, 1) as (1, N)
    #     if self.verbose: print('greedy_decode: ys', ys)
    #     for i in range(max_len-1):
    #         memory = memory.to(self.device) # memory.shape = (S, 1, E) as N=1
    #         tgt_mask = (self.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
    #         if self.verbose: print(f'greedy_decode: memory.shape: {memory.shape}\ttgt_mask.shape: {tgt_mask.shape}')
    #         out = self.decode(ys, memory, tgt_mask) # out.shape = (1, 1, E) as (T, N, E)
    #         out = out.transpose(0, 1) # out.shape = (1, 1, E) as (N, T, E)
    #         if self.verbose: print(f'greedy_decode: out.shape: {out.shape}\texpect: (T,N,E) --> (N,T,E)')
    #         prob = self.generator(out[:, -1, :]) # prob.shape = (1, out_dim) as (N, out_dim)
    #         if self.verbose: print(f'greedy_decode: pred prob.shape: {prob.shape}\texpect: (N, T, out_dim)')
    #         _, next_word = torch.max(prob, dim=1)
    #         next_word = next_word.item()
    #         ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    #         if next_word == EOS_ID: break
    #     if self.verbose: print(f'greedy_decode: final ys.shape: {ys.shape}')
    #     return ys # ys.shape = (T, 1) as (T, N)

    # # actual function to translate input sentence into target language
    # def translate(self, src_sentence: str, tkzer_dict, field_dict):
    #     # FOR INFERENCE
    #     self.eval()
    #     tkzer_src = tkzer_dict[self.cfg['seq2seq']['model_lang']['in_lang']]
    #     field_src = field_dict[self.cfg['seq2seq']['model_lang']['in_lang']]
    #     field_trg = field_dict[self.cfg['seq2seq']['model_lang']['out_lang']]
    #     # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    #     src, src_len_tensor = sent2tensor(tkzer_src, field_src, field_trg, self.device, 64, src_sentence)
    #     # src.shape = (S, 1)
    #     if self.verbose: print(f'translate: src.shape: {src.shape}, src: {src}')
    #     num_tokens = src.shape[0]
    #     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    #     tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_ID).flatten()
    #     # return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    #     results = idx2sent(field_trg, tgt_tokens.unsqueeze(1))[0]
    #     # results = ' '.join(results).replace("<sos>", "").replace("<eos>", "").split()
    #     return {'results': results, 'tokens': tgt_tokens}