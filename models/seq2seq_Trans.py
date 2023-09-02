# https://www.kaggle.com/code/latinchakma/transformer-based-seq2seq-de-to-en-multi30k
# check hq.nguyen1115@gmail.com for latest ver.

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
    def __init__(self, cfg, in_lang, out_lang, src_pad_idx, device):
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
        src = src.to(self.device)
        trg = trg.to(self.device)

        trg_in = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, trg_in)
        
        memory_key_padding_mask = src_padding_mask # OR = NONE ???

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg_in))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # outs = self.decode(tgt_emb, self.encode(src, src_mask), tgt_mask)
        # outs.shape = [seq_len, batch_size, dim_feedforward]

        logits = self.generator(outs)
        # return logits

        if criterion != None:
          loss = self.compute_loss(logits, trg, criterion)
          return loss, logits, None
        return logits, None

    def encode(self, src: Tensor, src_mask: Tensor):
        # return: enc.shape = [seq_len, batch, emb_size]
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
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
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == PAD_ID).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_ID).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def prep_input(self, batch, model_cfg):
        '''
        batch: dict of langs. each lang is tuple (seq_batch, seq_lens)
        '''
        # return (
        # batch[SRC_LANGUAGE],
        # batch[TGT_LANGUAGE]
        # )
        return (
          batch[model_cfg['seq2seq']['model_lang']['in_lang']],
          batch[model_cfg['seq2seq']['model_lang']['out_lang']]
        )

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src, src_mask = src.to(self.device), src_mask.to(self.device)

        memory = self.encode(src, src_mask)
        print('src.shape', src.shape, 'src_mask.shape', src_mask.shape, 'memory.shape', memory.shape)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_ID:
                break
        return ys

    # actual function to translate input sentence into target language
    def translate(self, src_sentence: str, tkzer_src, field_src, field_trg):
        self.eval()
        # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        src, src_len_tensor = sent2tensor(tkzer_src, field_src, field_trg, self.device, 64, src_sentence)

        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5, start_symbol=SOS_ID).flatten()
        # return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        results = idx2sent(field_trg, tgt_tokens.unsqueeze(1))[0]
        results = ' '.join(results).replace("<sos>", "").replace("<eos>", "").split()
        return results