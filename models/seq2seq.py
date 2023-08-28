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

from .networks import EncoderRNN, DecoderRNN, AttentionRNN
from .model_utils import init_weights

class Seq2SeqRNN(nn.Module):
  def __init__(self, cfg, in_lang, out_lang, src_pad_idx, device):
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('piv', '')
    self.cfg.pop('tri', '')
    self.cfg['seq2seq']['model_lang'] = self.model_lang = {"in_lang": in_lang, "out_lang": out_lang}
    self.cfg['model_id'] = self.modelname = 'RNN_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])

    attn = AttentionRNN(cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['HID_DIM'])
    self.encoder = EncoderRNN(cfg['seq2seq'][f'{in_lang}_DIM'], cfg['seq2seq']['EMB_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['DROPOUT'])
    self.decoder = DecoderRNN(cfg['seq2seq'][f'{out_lang}_DIM'], cfg['seq2seq']['EMB_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['DROPOUT'], attn)

    self.src_pad_idx = src_pad_idx
    self.device = device

    os.makedirs(self.save_dir, exist_ok=True)
    self.apply(init_weights)

  def create_mask(self, src):
    mask = (src != self.src_pad_idx).permute(1, 0)
    return mask

  def forward(self, batch, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    # batch: dict of langs
    #src = [src len, batch size]
    #src_len = [batch size]
    #trg = [trg len, batch size]
    #trg_len = [batch size]
    #teacher_forcing_ratio is probability of using trg to be input else prev output to be input for next prediction.
    (src, src_len), (trg, _) = self.prep_input(batch, model_cfg)
    print('seq2seq', model_cfg['seq2seq']['model_lang']['in_lang'], model_cfg['seq2seq']['model_lang']['out_lang'])
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    # SORT
    sort_ids, unsort_ids = self.sort_by_sent_len(src_len)
    src, src_len, trg = src[:, sort_ids], src_len[sort_ids], trg[:, sort_ids]

    #tensor to store decoder outputs
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

    #encoder_outputs is all hidden states of the input sequence, back and forwards
    #hidden is the final forward and backward hidden states, passed through a linear layer
    encoder_outputs, hidden = self.encoder(src, src_len)

    #first input to the decoder is the <sos> tokens
    input = trg[0,:]

    mask = self.create_mask(src)  #mask = [batch size, src len]

    for t in range(1, trg_len):
      #insert input token embedding, previous hidden state, all encoder hidden states and mask
      #receive output tensor (predictions) and new hidden state
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

      #place predictions in a tensor holding predictions for each token
      outputs[t] = output

      #if teacher forcing, use actual next token as next input. Else, use predicted token
      input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1)

    if criterion != None:
      loss = self.compute_loss(outputs, trg, criterion)
      return loss, outputs[:, unsort_ids, :]
    return outputs[:, unsort_ids, :]

  def compute_loss(self, output, trg, criterion):
    #output = (trg_len, batch_size, trg_vocab_size)
    #trg = [trg len, batch size]
    output = output[1:].view(-1, output.shape[-1])  #output = [(trg len - 1) * batch size, output dim]
    trg = trg[1:].view(-1)  #trg = [(trg len - 1) * batch size]
    loss = criterion(output, trg)
    return loss

  def sort_by_sent_len(self, sent_len):
    _, sort_ids = sent_len.sort(descending=True)
    unsort_ids = sort_ids.argsort()
    return sort_ids, unsort_ids

  def prep_input(self, batch, model_cfg):
    '''
    batch: dict of langs. each lang is tuple (seq_batch, seq_lens)
    '''
    return (
      batch[model_cfg['seq2seq']['model_lang']['in_lang']],
      batch[model_cfg['seq2seq']['model_lang']['out_lang']]
    )