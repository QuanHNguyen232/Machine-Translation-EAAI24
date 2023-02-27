"""
@Creator: Quan Nguyen
@Date: Feb 27, 2023
@Credits: Quan Nguyen

util.py file for utils
"""

import random
import math
import time
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import Field

from baseattn import Seq2Seq

class PivotSeq2Seq(nn.Module):
  def __init__(self, model1: Seq2Seq, model2: Seq2Seq,
               src_field: Field, piv_field: Field, trg_field: Field,
               device, lamda=0.75):
    super().__init__()
    self.model1 = model1
    self.model2 = model2
    self.src_field = src_field
    self.piv_field = piv_field
    self.trg_field = trg_field
    self.device = device
    self.lamda = lamda
      
  def forward(self, src, src_len, piv, piv_len, trg, criterion=None, teacher_forcing_ratio=0.5):
    #src = [src len, batch size]
    #src_len = [batch size]
    #piv = [piv len, batch size]
    #trg = [trg len, batch size]
    if criterion != None:
      criterion1, criterion2 = criterion
      loss1, output1 = self.model1(src, src_len, piv, criterion1, teacher_forcing_ratio) # output1 = [piv len, batch size, output dim]

      if random.random() < teacher_forcing_ratio:
        piv, piv_len, trg = self.process_output(output1, trg) # piv = [piv len, batch_size] & piv_len = [batch_size]
      else:
        piv, piv_len, trg = self.sort_by_piv(piv, piv_len, trg)

      loss2, output2 = self.model2(piv, piv_len, trg, criterion2, teacher_forcing_ratio) # output2 = [trg len, batch size, output dim]
      return self.compute_loss(loss1, loss2), output2
    
    output1 = self.model1(src, src_len, piv, teacher_forcing_ratio)
    if random.random() < teacher_forcing_ratio:
      piv, piv_len, trg = self.process_output(output1, trg)
    else:
      piv, piv_len, trg = self.sort_by_piv(piv, piv_len, trg)
    output2 = self.model2(piv, piv_len, trg, teacher_forcing_ratio)

    return output2
  
  def compute_loss(self, loss1, loss2):
    return loss1 + loss2 + self.lamda*self.compute_embed_loss()
  
  def compute_embed_loss(self):
    # decoder.embedding.weight = [piv_vocab_dim, emb_dim]
    # F.pairwise_distance = [piv_vocab_dim] --> distance for each vector along emb_dim
    return torch.sum(F.pairwise_distance(self.model1.decoder.embedding.weight, self.model2.encoder.embedding.weight, p=2))
  
  def sort_by_piv(self, piv, piv_len, trg):
    piv_len, sorted_ids = piv_len.sort(descending=True)
    return piv[:, sorted_ids], piv_len, trg[:, sorted_ids]

  def process_output(self, output, trg):
    # output = [trg len, batch size, output dim]
    # trg = [trg len, batch size]
    # Process output1 to be input for model2
    seq_len, N, _ = output.shape
    tmp_out = output.argmax(2)  # tmp_out = [seq_len, batch_size]
    # re-create piv as src for model2
    piv = torch.zeros_like(tmp_out).type(torch.long).to(output.device)
    piv[0, :] = torch.full_like(piv[0, :], self.piv_field.vocab.stoi[self.piv_field.init_token])  # fill all first idx with sos_token
    
    for i in range(1, tmp_out.shape[0]):  # for each i in seq_len
      # if tmp_out's prev is eos_token, replace w/ pad_token, else current value
      eos_mask = tmp_out[i-1, :] == self.piv_field.vocab.stoi[self.piv_field.eos_token]
      piv[i, :] = torch.where(eos_mask, self.piv_field.vocab.stoi[self.piv_field.pad_token], tmp_out[i, :])
      # if piv's prev is pad_token, replace w/ pad_token, else current value
      pad_mask = (piv[i-1, :] == self.piv_field.vocab.stoi[self.piv_field.pad_token])
      piv[i, :] = torch.where(pad_mask, self.piv_field.vocab.stoi[self.piv_field.pad_token], piv[i, :])
    
    # Trim down extra pad tokens
    tensor_list = [piv[i] for i in range(seq_len) if not all(piv[i] == self.piv_field.vocab.stoi[self.piv_field.pad_token])]  # tensor_list = [new_seq_len, batch_size]
    piv = torch.stack([x for x in tensor_list], dim=0).type(torch.long).to(output.device)
    
    # get seq_id + eos_tok id of each sequence
    piv_ids, eos_ids = (piv.permute(1, 0) == self.piv_field.vocab.stoi[self.piv_field.eos_token]).nonzero(as_tuple=True)  # piv_len = [N]
    piv_len = torch.full_like(piv[0], piv.shape[0]).type(torch.long)  # fill w/ seq_len in case no <eos> found
    piv_len[piv_ids] = eos_ids + 1 # include eos tok in len # still works even when piv_ids, piv_len = tensor([], dtype=...)
      
    assert not all(piv[-1] == self.piv_field.vocab.stoi[self.piv_field.pad_token]), 'Not completely trim down tensor'

    # piv = [seq_len, batch_size]
    piv, piv_len, trg = self.sort_by_piv(piv, piv_len, trg)
    return piv, piv_len, trg