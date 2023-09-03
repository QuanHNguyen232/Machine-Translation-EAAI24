"""
@Creator: Quan Nguyen
@Date: Aug 21st, 2023
@Credits: Quan Nguyen
"""

import os
import copy
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Transformer

from .seq2seq_trans import Seq2SeqTransformer
from .model_utils import init_weights

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

class PivotSeq2SeqMultiSrc(nn.Module):
  def __init__(self, cfg, models: list, device):
    super().__init__()

class TriangSeq2SeqMultiSrc(nn.Module):
  def __init__(self, cfg, models: list, device):
    super().__init__()
    # output_dim = trg vocab size
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('seq2seq', '')
    self.cfg.pop('piv', '')
    self.cfg['model_id'] = self.modelname = 'triMultiSrcTrans_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])

    self.is_train_backbone = cfg['tri']['is_train_backbone']
    self.out_lang = 'fr'
    self.output_dim = cfg['seq2seq']['fr_DIM']
    self.device = device

    self.num_model = len(models)
    # self.submodels = []
    self.piv_langs = []
    self.add_submodels(models, cfg)
    self.decoder = Seq2SeqTransformer(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device).decode_train
    self.fc = nn.Linear(cfg['seq2seq']['FFN_HID_DIM'] * (self.num_model + 1), cfg['seq2seq']['FFN_HID_DIM'])

    # os.makedirs(self.save_dir, exist_ok=True)
    # self.apply(init_weights)

  def add_submodels(self, models: list, cfg):
    for i, submodel in enumerate(models):
        # validity check
        assert isinstance(submodel, Seq2SeqTransformer), f'{type(submodel)} != Seq2SeqTransformer'
        # add submodel
        for param in submodel.parameters():
            param.requires_grad = self.is_train_backbone

        # self.submodels.append(submodel)
        self.add_module(f'submodel_{i}', submodel)
        self.cfg['tri'][f'model_{i}'] = submodel.cfg
        enc_in_lang = submodel.cfg['seq2seq']['model_lang']['out_lang']
        print('submodel in_lang', enc_in_lang)
        self.piv_langs.append(submodel.cfg['seq2seq']['model_lang']['out_lang'])
        self.add_module(f'enc_{i}', Seq2SeqTransformer(cfg, enc_in_lang, 'fr', PAD_ID, self.device))
    # for ENG
    self.add_module(f'enc_en', Seq2SeqTransformer(cfg, 'en', 'fr', PAD_ID, self.device))

  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    loss_list, output_list, output_preds = self.run(batch, model_cfg, criterion, teacher_forcing_ratio)
    logits = self.get_final_pred_train(batch, output_preds, teacher_forcing_ratio)
    if criterion != None:
      total_loss = self.compute_submodels_loss(loss_list) + self.compute_final_pred_loss(logits, batch[self.out_lang], criterion)
      total_loss /= (len(loss_list) + 1)
      print('TriangSeq2SeqMultiSrc FORWARD')
      return total_loss, logits, None
    else:
      print('TriangSeq2SeqMultiSrc FORWARD')
      return logits, None

  def get_output_pred(self, submodel, submodel_cfg, batch):
    (src, _), (trg, _) = submodel.prep_input(batch, submodel_cfg)
    num_tokens = src.shape[0]
    trg_in = trg[:-1, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, trg_in)
    output_pred = submodel.greedy_decode(src, src_mask, num_tokens + 5, SOS_ID)
    return output_pred

  def run(self, batch, model_cfg, criterion, teacher_forcing_ratio):
    '''
    for En -> piv_lang models (direct models)
    '''
    loss_list, output_list, output_preds = [], [], []
    for i in range(self.num_model):
      # GET MODEL
      # submodel = self.submodels[i] # getattr(self, f'model_{i}')
      submodel = getattr(self, f'submodel_{i}')
      submodel_cfg = model_cfg['tri'][f'model_{i}']
      # FORWARD MODEL
      output = submodel(batch, submodel_cfg, criterion, 0 if criterion==None else teacher_forcing_ratio)
      output_pred = self.get_output_pred(submodel, submodel_cfg, batch)
      output_preds.append(output_pred)
      
      if criterion != None:
        loss_list.append(output[0])
        output_list.append(output[1])
      else:
        output_list.append(output[0])
    print('TriangSeq2SeqMultiSrc RUN')
    return loss_list, output_list, output_preds

  def compute_submodels_loss(self, loss_list):
    total_loss = 0.0
    for loss in loss_list:
      total_loss += loss
    return total_loss

  def compute_final_pred_loss(self, logits, data, criterion):
    #output = (trg_len, batch_size, trg_vocab_size)
    #data = [trg, trg_len]  # trg.shape = [seq_len, batch_size]
    tgt, _ = data
    tgt_out = tgt[1:, :]
    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    return loss
  
  def get_encOut_hid(self, encname, src, trg):
    '''
    return tgt_emb, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask
    '''
    # for each output, feed through enc to get enc_output, hidden
    print('get_encOut_hid get encname', encname)
    encoder_output = getattr(self, encname).encode_train(src, trg)
    return encoder_output
  
  def get_combined_memory(self, encs):
    memory_list = [enc['memory'] for enc in encs] # memory = (S, N, E)
    return self.fc(torch.cat(memory_list, dim=2))
  
  def get_combined_memory_key_padding_mask(self, encs):
    # combine = ~torch.logical_or(~src_padding_mask1, ~src_padding_mask2) # ~: invert
    memory_key_padding_mask_list = [enc['memory_key_padding_mask'] for enc in encs]
    combine_memory_key_padding_mask = memory_key_padding_mask_list[0]
    for i in range(1, len(memory_key_padding_mask_list)):
      combine_memory_key_padding_mask = ~torch.logical_or(
        ~combine_memory_key_padding_mask, ~memory_key_padding_mask_list[i])
    return combine_memory_key_padding_mask

  def get_final_pred_train(self, batch, output_list, teacher_forcing_ratio):
    # output_list[0] shape = [seq_len, N, out_dim], each output is in a language
    en_tensor, en_tensor_len = batch['en']
    (trg, _) = batch[self.out_lang]
    trg_len, batch_size = trg.shape
    trg_vocab_size = self.decoder.output_dim

    # Padd all srcs 
    srcs = []
    len_output_list = len(output_list)
    for i, output in enumerate(output_list):
      src, _ = batch[self.piv_langs[i]] if random.random() < teacher_forcing_ratio else self.process_output(output)
      srcs.append(src) # src = (S, N)
    pad_srcs = torch.nn.utils.rnn.pad_sequence(srcs) # src = (S, new_dim, N)
    output_list = [pad_srcs[:, i, :] for i in range(len_output_list)] # shape = (S, N)

    encs = []
    for i, src in enumerate(output_list):
      print('get_final_pred lang =', self.piv_langs[i])
      encoder_output = self.get_encOut_hid(f'enc_{i}', src, trg)
      encs.append(encoder_output)

    # FOR ENG
    encoder_output = self.get_encOut_hid(f'enc_en', en_tensor, trg)
    encs.append(encoder_output)

    # combine memory
    combine_memory = self.get_combined_memory(encs)

    # combine memory_key_padding_mask
    combine_memory_key_padding_mask = self.get_combined_memory_key_padding_mask(encs)

    # encoder_output = self.encode_train(src, trg)
    x = {
      "tgt_emb": encs[0]['tgt_emb'],
      "memory": combine_memory,
      "tgt_mask": encs[0]['tgt_mask'],
      "tgt_padding_mask": encs[0]['tgt_padding_mask'],
      "memory_key_padding_mask": combine_memory_key_padding_mask
    }
    logits = self.decode_train(**x)

    return logits
  
  def greedy_decode(self, src, src_mask, max_len, start_symbol):
    pass
  def translate(self, src_sentences, tkzer_dict, field_dict):
    self.eval()

  def process_output(self, output): # from PIV
    # output = [trg len, batch size, output dim]
    # trg = [trg len, batch size]
    # Process output1 to be input for model2
    seq_len, N, _ = output.shape
    tmp_out = output.argmax(2)  # tmp_out = [seq_len, batch_size]
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