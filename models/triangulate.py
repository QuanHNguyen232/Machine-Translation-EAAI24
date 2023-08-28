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
from .pivot import PivotSeq2Seq
from .model_utils import init_weights

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

class TriangSeq2Seq(nn.Module):
  def __init__(self, cfg, models: list, device):
    # output_dim = trg vocab size
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('seq2seq', '')
    self.cfg.pop('piv', '')
    self.cfg['model_id'] = self.modelname = 'tri_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])

    self.alpha = cfg['tri']['alpha']
    self.method = cfg['tri']['method']
    self.is_train_backbone = cfg['tri']['is_train_backbone']
    self.output_dim = cfg['seq2seq']['fr_DIM']

    self.num_model = len(models)
    self.add_submodels(models)

    self.device = device

    if self.method=='weighted':
      self.head = nn.Sequential(
          nn.ReLU(),
          nn.Linear(self.output_dim*self.num_model, self.output_dim)
      )
    elif self.method=='weighted_1':
      self.head = nn.Sequential(  # traing-EnFr-EnEsFr-dense5
            nn.Linear(self.num_model, 5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(5, 1),
        )

    os.makedirs(self.save_dir, exist_ok=True)
    self.apply(init_weights)

  def add_submodels(self, models: list):
    for i, submodel in enumerate(models):
      # validity check
      assert isinstance(submodel, Seq2SeqRNN) or isinstance(submodel, PivotSeq2Seq), f'{type(submodel)} != Seq2SeqRNN or PivotSeq2Seq'
      # add submodel
      for param in submodel.parameters(): param.requires_grad = self.is_train_backbone
      self.add_module(f'model_{i}', submodel)
      self.cfg['tri'][f'model_{i}'] = submodel.cfg

  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    '''
    batch: dict of data:
      {"model_0": (src, src_len, trg, trg_len), "model_1": [(src, src_len), (piv, piv_len), (trg, trg_len)], ..., "TRG": (trg, trg_len)}
      src = [src len, batch size]
      src_len = [batch size]
    '''
    loss_list, output_list = self.run(batch, model_cfg, criterion, teacher_forcing_ratio)
    final_out = self.get_final_pred(output_list)
    if criterion != None:
      total_loss = self.compute_submodels_loss(loss_list)
      if self.method != 'max': # why non "max"? Is it because gradient cannot pass through that method?
        total_loss += self.alpha*self.compute_final_pred_loss(final_out, batch["fr"], criterion)
      return total_loss, final_out
    else:
      return final_out

  def run(self, batch, model_cfg, criterion, teacher_forcing_ratio):
    loss_list, output_list = [], []
    for i in range(self.num_model):
      submodel = getattr(self, f'model_{i}')
      submodel_cfg = model_cfg['tri'][f'model_{i}']
      print('tri', f'model_{i}', submodel_cfg)
      output = submodel(batch, submodel_cfg, criterion, 0 if criterion==None else teacher_forcing_ratio)

      if criterion == None:
        output_list.append(output)
      else:
        assert len(output) == 2, 'With criterion, model should return loss & prediction'
        loss_list.append(output[0])
        output_list.append(output[1])

    return loss_list, output_list

  def compute_submodels_loss(self, loss_list):
    total_loss = 0.0
    for loss in loss_list:
      total_loss += loss
    return total_loss

  def compute_final_pred_loss(self, output, data, criterion):
    #output = (trg_len, batch_size, trg_vocab_size)
    #data = [trg, trg_len]  # trg.shape = [seq_len, batch_size]
    trg, _ = data
    output = output[1:].reshape(-1, output.shape[-1])  #output = [(trg len - 1) * batch size, output dim]
    trg = trg[1:].reshape(-1)  #trg = [(trg len - 1) * batch size]
    loss = criterion(output, trg)
    return loss

  def get_final_pred(self, output_list):  # output_list[0] shape = [seq_len, N, out_dim]
    # assert all([output_list[i].shape == output_list[i-1].shape for i in range(1, len(output_list))]), 'all outputs must match shape [seq_len, N, out_dim]'
    seq_len, N, out_dim = output_list[0].shape
    if self.method=='weighted':
      linear_in = torch.cat([out for out in output_list], dim=-1) # linear_in = [seq_len, N, out_dim * num_model]. Note that num_model = len(output_list)
      final_out = self.head(linear_in)  # final_out = [seq_len, N, out_dim]
      return final_out
    elif self.method=='weighted_1':
      output_list = [out.permute(1, 0, 2).reshape(N, -1) for out in output_list]  # [N, seq_len, out_dim] --> [N, seq_len*out_dim]
      final_out = self.head(torch.stack(output_list, dim=-1)).squeeze(-1)  # [N, seq_len*out_dim, num_model] --> [N, seq_len*out_dim, 1] --> [N, seq_len*out_dim]
      return final_out.reshape(N, seq_len, out_dim).permute(1, 0, 2) # [N, seq_len, out_dim] --> [seq_len, N, out_dim]
    elif self.method=='average':
      outputs = torch.mean(torch.stack(output_list, dim=0), dim=0)
      return outputs
    elif self.method=='max':
      all_t = torch.stack(output_list, dim=-1)
      prob_ts = torch.stack([F.softmax(d, -1) for d in output_list], dim=-1)

      final_selected_ts = []
      for sent_id in range(N):
        # get ids of each model (get selected words)
        ids_list = []
        for m in range(self.num_model):
          m_t = prob_ts[..., m] # [seq_len, N, out_dim]
          ids_list.append(torch.argmax(m_t[:, sent_id, :], dim=-1, keepdim=True))
        # get the confusion matrix
        all_pairs = []
        for t in range(self.num_model):
          t_eachM = []
          for m in range(self.num_model):
            m_t = prob_ts[..., m]
            t_eachM.append(torch.gather(m_t[:, sent_id, :], -1, ids_list[t]))
          all_pairs.append(t_eachM)
        # calculate the prob of each sent
        t_m_prod = []
        for t in range(self.num_model):
          t_eachM = []
          for m in range(self.num_model):
            t1_m1 = all_pairs[t][m]
            t_eachM.append(torch.mean(t1_m1, dim=0))  # original: prod
          t_m_prod.append(t_eachM)
        t_totals = [torch.stack(t_m_prod[t], dim=-1)for t in range(self.num_model)]
        t_means = [torch.mean(t_totals[t], dim=-1) for t in range(self.num_model)]
        t_cats = torch.cat(t_means, dim=-1)
        t_selects = torch.argmax(t_cats)
        selected_t = torch.select(all_t[:, sent_id, ...], -1, t_selects)
        final_selected_ts.append(selected_t)
      output = torch.stack(final_selected_ts, dim=1)
      return output
    else:
      return output_list[0]

class TriangSeq2SeqMultiSrc(TriangSeq2Seq):
  def __init__(self, cfg, models: list, device):
    super().__init__(cfg, models, device)