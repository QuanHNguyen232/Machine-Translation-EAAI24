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

from .seq2seq import Seq2SeqRNN

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

class PivotSeq2Seq(nn.Module):
  def __init__(self, cfg, models: list, device):
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('tri', '')
    self.cfg['piv']['model_lang'] = []
    self.cfg['model_id'] = self.modelname = 'piv_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])
    
    os.makedirs(self.save_dir, exist_ok=True)

    self.num_model = len(models)
    self.add_submodels(models)
    if self.cfg['piv']['is_share_emb']:
      self.set_share_emb()

    self.device = device

  def add_submodels(self, models: list):
    # validity check
    for i in range(1, self.num_model):
      assert models[i-1].out_lang == models[i].in_lang, f'{models[i-1].out_lang} != {models[i].in_lang}'
      assert isinstance(models[i-1], Seq2SeqRNN), f'{type(models[i-1])} != Seq2SeqRNN'
      assert isinstance(models[i], Seq2SeqRNN), f'{type(models[i])} != Seq2SeqRNN'
    # add submodel
    for i, submodel in enumerate(models):
      self.add_module(f'model_{i}', submodel)
      self.cfg['piv']['model_lang'].append((submodel.in_lang, submodel.out_lang))

  def set_share_emb(self):
    for i in range(1, self.num_model):
      prev_model = getattr(self, f'model_{i-1}')
      curr_model = getattr(self, f'model_{i}')
      curr_model.encoder.embedding = prev_model.decoder.embedding

  def forward(self, batch, criterion=None, teacher_forcing_ratio=0.5):
    if criterion != None:
      loss_list, output_list = self.run(batch, criterion, teacher_forcing_ratio)
      total_loss = self.compute_loss(loss_list)
      return total_loss, output_list[-1]
    else:
      _, output_list = self.run(batch, criterion, teacher_forcing_ratio)
      return output_list[-1]

  def run(self, batch, criterion, teacher_forcing_ratio):
    output_list, loss_list = [], []
    for i in range(self.num_model):
      # i==0: 1st model must always use src
      isForceOn = True if i==0 else random.random() < teacher_forcing_ratio

      # GET MODEL
      model = getattr(self, f'model_{i}') # Seq2Seq model already sort src by src_len in forward

      # GET NEW INPUT if needed
      if not isForceOn: # use prev_model's output as curr_model input
        (src, src_len), (_, _) = model.prep_input(batch)
        src, src_len = self.process_output(output_list[-1])
        batch[model.in_lang] = (src, src_len)

      # FORWARD MODEL
      # data = [(src, src_len), (trg, trg_len)]
      output = model(batch, criterion, 0 if criterion==None else teacher_forcing_ratio)

      if criterion == None:
        output_list.append(output)
      else:
        assert len(output) == 2, 'With criterion, model should return loss & prediction'
        loss, out = output
        loss_list.append(loss)
        output_list.append(out)

    return loss_list, output_list

  def compute_loss(self, loss_list:list):
    total_loss = 0.0
    for loss in loss_list: # except final output
      total_loss += loss
    total_loss /= len(loss_list)
    if self.cfg['piv']['is_share_emb']:
      total_loss += self.cfg['piv']['lambda']*self.compute_embed_loss()
    return total_loss

  def compute_embed_loss(self):
    embed_loss = 0.0
    for i in range(1, self.num_model):
      model1 = getattr(self, f'model_{i-1}')
      model2 = getattr(self, f'model_{i}')
      embed_loss += torch.sum(F.pairwise_distance(model1.decoder.embedding.weight, model2.encoder.embedding.weight, p=2))
    return embed_loss

  def sort_by_src_len(self, piv, piv_len, datas): # piv = [piv_len, batch_size]
    piv_len, sorted_ids = piv_len.sort(descending=True)
    sorted_datas = [(sent[:, sorted_ids], sent_len[sorted_ids]) for (sent, sent_len) in datas]
    return piv[:, sorted_ids], piv_len, sorted_datas  # piv sorted along batch_size

  def process_output(self, output):
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