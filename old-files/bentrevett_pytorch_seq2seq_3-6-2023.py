# -*- coding: utf-8 -*-
"""bentrevett_pytorch_seq2seq-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EPBv4kOFgzZNaaVSewqGiFTFXWuLmAgE

NEW TASKS:
* [X] Seq2Seq: sort by src_len and unsort output --> ensure output matches with trg
* [ ] Pivot model: ensure it works for $n$ seq2seq models
* [ ] Trian model: ensure outputs from all submodels match
"""

# piv_endefr_74kset_2.pt using PivotModel in bentrevett/pytorch-seq2seq-OLD.ipynb

# https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb
# based on https://gmihaila.github.io/tutorial_notebooks/pytorchtext_bucketiterator/#dataset-class

"""# Setup"""

!pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111 -q

import torch
print(torch.__version__)

!pip install torchtext==0.9 -q

!python -m spacy download en_core_web_sm -q
!python -m spacy download de_core_news_sm -q
!python -m spacy download fr_core_news_sm -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.data import Dataset, Example
from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import pickle
from tqdm import tqdm

"""# My Section

## Setup
"""

from google.colab import drive
drive.mount('/content/gdrive')

spacy_de = spacy.load('de_core_news_sm')
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

spacy_fr = spacy.load('fr_core_news_sm')
def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

spacy_en = spacy.load('en_core_web_sm')
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

EN_FIELD = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            include_lengths = True
            )
DE_FIELD = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            include_lengths = True
            )
FR_FIELD = Field(tokenize = tokenize_fr,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            include_lengths = True
            )

"""## Data"""

def save_vocab(vocab, path):
  with open(path, 'w+', encoding='utf-8') as f:
    for token, index in vocab.stoi.items():
      f.write(f'{index}\t{token}\n')
def read_vocab(path):
  vocab = dict()
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      index, token = line.split('\t')
      vocab[token] = int(index)
  return vocab
# https://discuss.pytorch.org/t/how-to-save-and-load-torchtext-data-field-build-vocab-result/50407/3
# save_vocab(TRG_FIELD.vocab, '/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/seq2seq-enfr-trg_vocab.txt')
# LOAD_FIELD.vocab = read_vocab('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/seq2seq-enfr-src_vocab.txt')

with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/enfr_160kpairs_2k5-freq-words.pkl', 'rb') as f:
  data = pickle.load(f)
data[80], len(data)

# with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/endefr_75kpairs_2k5-freq-words.pkl', 'rb') as f:
#   data = pickle.load(f)
# data[8], len(data)

train_len = 64000
valid_len = 3200
test_len = 6400

train_pt = train_len
valid_pt = train_pt + valid_len
test_pt = valid_pt + test_len

# For 2 langs
data_set = [[pair['en'].lower(), pair['fr'].lower()] for pair in data]
FIELDS = [('en', EN_FIELD), ('fr', FR_FIELD)]
train_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[: train_pt]))
valid_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[train_pt : valid_pt]))
test_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[valid_pt : test_pt]))

# For 3 langs
# data_set = [[pair['en'], pair['de'], pair['fr']] for pair in data]
# FIELDS = [('en', EN_FIELD), ('de', DE_FIELD), ('fr', FR_FIELD)]
# train_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[: train_pt]))
# valid_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[train_pt : valid_pt]))
# test_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[valid_pt : test_pt]))

train_dt = Dataset(train_examples, fields=FIELDS)
valid_dt = Dataset(valid_examples, fields=FIELDS)
test_dt = Dataset(test_examples, fields=FIELDS)

EN_FIELD.build_vocab(train_dt, min_freq = 1) # choose 1 since data is already filter w/ most freq words
DE_FIELD.build_vocab(train_dt, min_freq = 1)
FR_FIELD.build_vocab(train_dt, min_freq = 1)
len(EN_FIELD.vocab), len(DE_FIELD.vocab), len(FR_FIELD.vocab)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dt, valid_dt, test_dt),
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.en),
     device = device)

for i, batch in enumerate(train_iterator):
  print(batch.en[0].shape, batch.en[1])
  # print(batch.de[0].shape, batch.de[1])
  print(batch.fr[0].shape, batch.fr[1])
  if i==0: break
batch.fields

# batch.src

# batch.trg

# src_sent, piv_sent, trg_sent = [], [], []
# for i in batch.src[0][: , 0]:
#   src_sent.append(SRC_FIELD.vocab.itos[i])
# for i in batch.piv[0][: , 0]:
#   piv_sent.append(PIV_FIELD.vocab.itos[i])
# for i in batch.trg[0][:, 0]:
#   trg_sent.append(TRG_FIELD.vocab.itos[i])
# print(' '.join(src_sent))
# print(' '.join(piv_sent))
# print(' '.join(trg_sent))

# for i in range(5):
#   print(SRC_FIELD.vocab.itos[i], PIV_FIELD.vocab.itos[i], TRG_FIELD.vocab.itos[i])

# batch.src[0][:, 0]

"""## Model

### Encoder
"""

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, emb_dim)
    self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
    self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src, src_len):
    #src = [src len, batch size]
    #src_len = [batch size]
    embedded = self.dropout(self.embedding(src))  #embedded = [src len, batch size, emb dim]

    #need to explicitly put lengths on cpu!
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))

    #  when the input is a pad token are all zeros
    packed_outputs, hidden = self.rnn(packed_embedded)
    #packed_outputs is a packed sequence containing all hidden states
    #hidden is now from the final non-padded element in the batch

    outputs, len_list = nn.utils.rnn.pad_packed_sequence(packed_outputs) #outputs is now a non-packed sequence, all hidden states obtained
    #  when the input is a pad token are all zeros

    #outputs = [src len, batch size, hid dim * num directions]
    #hidden = [n layers * num directions, batch size, hid dim]

    #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
    #outputs are always from the last layer

    #hidden [-2, :, : ] is the last of the forwards RNN
    #hidden [-1, :, : ] is the last of the backwards RNN

    #initial decoder hidden is final hidden state of the forwards and backwards
    #  encoder RNNs fed through a linear layer
    hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

    #outputs = [src len, batch size, enc hid dim * 2]
    #hidden = [batch size, dec hid dim]
    return outputs, hidden

"""### Attn"""

class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super().__init__()
    self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
    self.v = nn.Linear(dec_hid_dim, 1, bias = False)

  def forward(self, hidden, encoder_outputs, mask):
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    batch_size = encoder_outputs.shape[1]
    src_len = encoder_outputs.shape[0]

    #repeat decoder hidden state src_len times
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  #hidden = [batch size, src len, dec hid dim]
    encoder_outputs = encoder_outputs.permute(1, 0, 2)  #encoder_outputs = [batch size, src len, enc hid dim * 2]
    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) #energy = [batch size, src len, dec hid dim]

    attention = self.v(energy).squeeze(2) #attention = [batch size, src len]
    attention = attention.masked_fill(mask == 0, -1e10)
    return F.softmax(attention, dim = 1)

"""### Decoder"""

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
    super().__init__()
    self.output_dim = output_dim
    self.attention = attention
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs, mask):
    #input = [batch size]
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    #mask = [batch size, src len]
    input = input.unsqueeze(0)  #input = [1, batch size]
    embedded = self.dropout(self.embedding(input))  #embedded = [1, batch size, emb dim]

    a = self.attention(hidden, encoder_outputs, mask) #a = [batch size, src len]
    a = a.unsqueeze(1)  #a = [batch size, 1, src len]

    encoder_outputs = encoder_outputs.permute(1, 0, 2)  #encoder_outputs = [batch size, src len, enc hid dim * 2]

    weighted = torch.bmm(a, encoder_outputs)  #weighted = [batch size, 1, enc hid dim * 2]
    weighted = weighted.permute(1, 0, 2)  #weighted = [1, batch size, enc hid dim * 2]

    rnn_input = torch.cat((embedded, weighted), dim = 2)  #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

    output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
    #output = [seq len, batch size, dec hid dim * n directions]
    #hidden = [n layers * n directions, batch size, dec hid dim]

    #seq len, n layers and n directions will always be 1 in this decoder, therefore:
    #output = [1, batch size, dec hid dim]
    #hidden = [1, batch size, dec hid dim]
    #this also means that output == hidden
    assert (output == hidden).all()

    embedded = embedded.squeeze(0)
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)

    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))  #prediction = [batch size, output dim]
    return prediction, hidden.squeeze(0), a.squeeze(1)

"""### Seq2Seq"""

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, src_pad_idx, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.device = device

  def create_mask(self, src):
    mask = (src != self.src_pad_idx).permute(1, 0)
    return mask

  def forward(self, datas, criterion=None, teacher_forcing_ratio = 0.5):
    #src = [src len, batch size]
    #src_len = [batch size]
    #trg = [trg len, batch size]
    #trg_len = [batch size]
    #teacher_forcing_ratio is probability of using trg to be input else prev output to be input for next prediction.
    (src, src_len), (trg, _) = datas
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

  # NEWLY ADDED ##########################
  def sort_by_sent_len(self, sent_len):
    _, sort_ids = sent_len.sort(descending=True)
    unsort_ids = sort_ids.argsort()
    return sort_ids, unsort_ids
  # END ADDED ############################

"""### Pivot model (update)

**still need to reorganize code to use for infer (no criterions, only 1 data: src, src_len)**
"""

class PivotSeq2Seq(nn.Module):
  def __init__(self, models: list, fields: list, device, lamda=0.75):
    super().__init__()
    self.num_model = len(models)
    self.fields = fields
    self.num_field = len(fields)
    self.device = device
    self.lamda = lamda

    for i in range(self.num_model):
      self.add_module(f'model_{i}', models[i])

    assert len(models)+1 == len(fields), f"Not enough Fields for models: num_field={len(fields)} != {len(models)+1}"

  def forward(self, datas: list, criterions=None, teacher_forcing_ratio=0.5):
    '''
    datas: list of data: [(src, src_len), (piv1, piv_len1), ... , (pivM, piv_lenM), (trg, trg_len)] given M models
      src = [src len, batch_size]
      src_len = [batch_size]
      ...
      trg = [trg len, batch_size]
      trg_len = [batch_size]
    criterions: list of criterion for each model
    '''
    if criterions != None:
      loss_list, output_list = self.run(datas, criterions, teacher_forcing_ratio)
      total_loss = self.compute_loss(loss_list)
      return total_loss, output_list[-1]
    else:
      criterions = [None for _ in range(self.num_model)]
      _, output_list = self.run(datas, criterions, teacher_forcing_ratio)
      return output_list[-1]

  def run(self, datas, criterions, teacher_forcing_ratio):
    assert self.num_model+1 == len(datas), f"Not enough datas for models: data_len={len(datas)} != {self.num_model+1}"
    assert self.num_model == len(criterions), f'Criterions must have for each model: num_criterion={len(criterions)} != {self.num_model}'

    output_list, loss_list = [], []
    for i in range(self.num_model):
      # 1st model must always use src
      isForceOn = True if i==0 else random.random() < teacher_forcing_ratio

      # GET NEW INPUT
      src, src_len = datas[i] if isForceOn else self.process_output(output_list[-1], self.fields[i+1])
      trg, trg_len = datas[i+1]

      # FORWARD MODEL
      model = getattr(self, f'model_{i}') # Seq2Seq model already sort src by src_len in forward
      data = [(src, src_len), (trg, trg_len)]
      criterion = criterions[i]
      output = model(data, criterion, 0 if criterion==None else teacher_forcing_ratio)

      if criterion == None:
        output_list.append(output)
      else:
        assert len(output) == 2, 'With criterion, model should return loss & prediction'
        loss, out = output
        loss_list.append(loss)
        output_list.append(out)

    return loss_list, output_list

  def compute_loss(self, loss_list):
    total_loss = 0.0
    for loss in loss_list:
      total_loss += loss
    return total_loss + self.lamda*self.compute_embed_loss()

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

  def process_output(self, output, piv_field):
    # output = [trg len, batch size, output dim]
    # trg = [trg len, batch size]
    # Process output1 to be input for model2
    seq_len, N, _ = output.shape
    tmp_out = output.argmax(2)  # tmp_out = [seq_len, batch_size]
    # re-create pivot as src for model2
    piv = torch.zeros_like(tmp_out).type(torch.long).to(output.device)
    piv[0, :] = torch.full_like(piv[0, :], piv_field.vocab.stoi[piv_field.init_token])  # fill all first idx with sos_token

    for i in range(1, seq_len):  # for each i in seq_len
      # if tmp_out's prev is eos_token, replace w/ pad_token, else current value
      eos_mask = (tmp_out[i-1, :] == piv_field.vocab.stoi[piv_field.eos_token])
      piv[i, :] = torch.where(eos_mask, piv_field.vocab.stoi[piv_field.pad_token], tmp_out[i, :])
      # if piv's prev is pad_token, replace w/ pad_token, else current value
      pad_mask = (piv[i-1, :] == piv_field.vocab.stoi[piv_field.pad_token])
      piv[i, :] = torch.where(pad_mask, piv_field.vocab.stoi[piv_field.pad_token], piv[i, :])

    # Trim down extra pad tokens
    tensor_list = [piv[i] for i in range(seq_len) if not all(piv[i] == piv_field.vocab.stoi[piv_field.pad_token])]  # tensor_list = [new_seq_len, batch_size]
    piv = torch.stack([x for x in tensor_list], dim=0).type(torch.long).to(output.device)
    assert not all(piv[-1] == piv_field.vocab.stoi[piv_field.pad_token]), 'Not completely trim down tensor'

    # get seq_id + eos_tok id of each sequence
    piv_ids, eos_ids = (piv.permute(1, 0) == piv_field.vocab.stoi[piv_field.eos_token]).nonzero(as_tuple=True)  # piv_len = [N]
    piv_len = torch.full_like(piv[0], seq_len).type(torch.long)  # init w/ longest seq
    piv_len[piv_ids] = eos_ids + 1 # seq_len = eos_tok + 1

    return piv, piv_len

"""### Triangulate model"""

class TriangSeq2Seq(nn.Module):
  def __init__(self, models: list, output_dim, device):
    # output_dim = trg vocab size
    super().__init__()
    self.num_model = len(models)
    self.device = device
    self.output_dim = output_dim
    self.final_fc = torch.nn.Linear(output_dim*len(models), output_dim)

    for i in range(self.num_model):
      self.add_module(f'model_{i}', models[i])

  def forward(self, datas: dict, criterions=None, teacher_forcing_ratio=0.5):
    '''
    datas: dict of data:
      {"model_0": (src, src_len, trg, trg_len), "model_1": [(src, src_len), (piv, piv_len), (trg, trg_len)], ..., "TRG": (trg, trg_len)}
      src = [src len, batch size]
      src_len = [batch size]
      ...
      trg = [trg len, batch size]
      trg_len = [batch size]
    criterions: dict of criterions
      {"model_0": criterion_0, "model_1": criterion_1, ..., "TRG": criterion_M}
    '''
    if criterions != None:
      loss_list, output_list = self.run(datas, criterions, teacher_forcing_ratio)
      assert len(loss_list)==len(output_list) and len(output_list)==self.num_model, 'DO NOT MATCH: len(loss_list)=len(output_list) OR len(output_list)=self.num_model'
      # TODO: calculate final_out from all outputs
      final_out = self.get_final_pred(output_list)
      # TODO: submodels_loss + final output's loss ==> total_loss
      total_loss = self.compute_final_pred_loss(final_out, datas["TRG"], criterions["TRG"]) + self.compute_submodels_loss(loss_list)
      return total_loss, final_out
    else:
      criterions = {f'model_{i}':None for i in range(self.num_model)}
      loss_list, output_list = self.run(datas, criterions, teacher_forcing_ratio)
      assert len(loss_list)==0 and len(output_list)==self.num_model, 'DO NOT MATCH: len(loss_list)=0 OR len(output_list)=self.num_model'
      # TODO: calculate final_out from all outputs
      final_out = self.get_final_pred(output_list)
      return final_out

  def run(self, datas, criterions, teacher_forcing_ratio):
    assert self.num_model+1 == len(datas), f"Not enough datas for models: data_len={len(datas)} != {self.num_model+1}"
    assert self.num_model == len(criterions), f'Criterions must have for each model: num_criterion={len(criterions)} != {self.num_model}'

    output_list = []
    loss_list = []
    for i in range(self.num_model):
      # 1st model must always use src
      isForceOn = True if i==0 else random.random() < teacher_forcing_ratio

      data = datas[f'model_{i}']
      model = getattr(self, f'model_{i}')
      criterion = criterions[f'model_{i}']
      output = model(data, criterion, 0 if criterion==None else teacher_forcing_ratio)

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
    return total_loss + self.lamda*self.compute_embed_loss()

  def compute_final_pred_loss(self, output, trg, criterion):
    #output = (trg_len, batch_size, trg_vocab_size)
    #trg = [trg len, batch size]
    output = output[1:].view(-1, output.shape[-1])  #output = [(trg len - 1) * batch size, output dim]
    trg = trg[1:].view(-1)  #trg = [(trg len - 1) * batch size]
    loss = criterion(output, trg)
    return loss

  def get_final_pred(self, output_list, method='weighted'):
    # output_list[0] shape = [seq_len, N, out_dim]
    # outputs must match shape because use the same trg, trg_len
    assert all([output_list[i].shape == output_list[i-1].shape for i in range(1, len(output_list))]), 'all outputs must match shape [seq_len, N, out_dim]'
    # MAX method: get max along seq_len b/w all outputs
    if method=='max':
      stack_dim = 2
      all_outputs = torch.stack([out for out in output_list], dim=stack_dim)  # all_outputs = [seq_len, N, stack_dim, out_dim]
      final_out, max_idx = torch.max(all_outputs, dim=stack_dim)  # final_out = [seq_len, N, out_dim]
      return final_out
    elif method=='weighted':
      linear_in = torch.cat([out for out in output_list], dim=-1) # linear_in = [seq_len, N, out_dim * num_model]. Note that num_model = len(output_list)
      final_out = self.final_fc(linear_in)  # final_out = [seq_len, N, out_dim]
      return final_out
    else:
      return output_list[0]

"""## Train func"""

def trainSeq2Seq(model, iterator, optimizer, criterion, clip):
  model.train()
  epoch_loss = 0.0
  for batch in tqdm(iterator):
    optimizer.zero_grad()
    datas = [batch.en, batch.fr]
    loss, output = model(datas, criterion)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
  return epoch_loss / len(iterator)

def evaluateSeq2Seq(model, iterator, criterion):
  model.eval()
  epoch_loss = 0.0
  with torch.no_grad():
    for batch in tqdm(iterator):
      datas = [batch.en, batch.fr]
      loss, output = model(datas, criterion, 0) # turn off teacher forcing
      epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def trainPivot(model, iterator, optimizer, criterions, clip):
  model.train()
  epoch_loss = 0.0
  for batch in tqdm(iterator):
    optimizer.zero_grad()
    model_inputs = [batch.en, batch.de, batch.fr]
    loss, output = model(model_inputs, criterions)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
  return epoch_loss / len(iterator)

def evaluatePivot(model, iterator, criterions):
  model.eval()
  epoch_loss = 0.0
  with torch.no_grad():
    for batch in tqdm(iterator):
      model_inputs = [batch.en, batch.de, batch.fr]
      loss, output = model(model_inputs, criterions, 0)
      epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# for triangulate model

# model.train()
# epoch_loss = 0.0
# iterator = train_iterator
# for batch in tqdm(iterator):
#   optimizer.zero_grad()
#   (en_sent, en_len), (de_sent, de_len), (fr_sent, fr_len) = batch.en, batch.de, batch.fr

#     # prep for model_1
#   _, sorted_ids = de_len.sort(descending=True)
#   model_inputs = {"model_0": [en_sent, en_len, fr_sent, fr_len],
#                   "model_1": [de_sent[:, sorted_ids], de_len[sorted_ids], fr_sent[:, sorted_ids], fr_len[sorted_ids]],
#                   "TRG": (fr_sent, fr_len)}
#   loss, output = model(model_inputs, criterions)

#   loss.backward()
#   torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#   optimizer.step()
#   epoch_loss += loss.item()
#   break

def update_trainlog(data: list, filename: str='/content/gdrive/MyDrive/Colab Notebooks/eaai24/training_log.txt'):
  ''' Update training log w/ new losses
  Args:
      data (List): a list of infor for many epochs as tuple, each tuple has model_name, loss, etc.
      filename (String): path + file_name
  Return:
      None: new data is appended into train-log
  '''
  with open(filename, 'a') as f: # save
    for epoch in data:
      f.write(','.join(epoch))
      f.write("\n")
  print('update_trainlog SUCCESS')
  return []

"""## Train"""

# # For 2 langs
INPUT_DIM = 2500  #len(SRC_FIELD.vocab) since vocab is selected from most 2k5 freq words
OUTPUT_DIM = 2500 #len(TRG_FIELD.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LR = 0.001
SRC_PAD_IDX = DE_FIELD.vocab.stoi[DE_FIELD.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

# For 3 langs
# INPUT_DIM = 2500  #len(SRC_FIELD.vocab) since vocab is selected from most 2k5 freq words
# PIV_DIM = 2500  #len(PIV_FIELD.vocab)
# OUTPUT_DIM = 2500 #len(TRG_FIELD.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5
# LR = 0.001

# SRC_PAD_IDX = EN_FIELD.vocab.stoi[EN_FIELD.pad_token]
# PIV_PAD_IDX = DE_FIELD.vocab.stoi[DE_FIELD.pad_token]

# attn1 = Attention(ENC_HID_DIM, DEC_HID_DIM)
# enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# dec1 = Decoder(PIV_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn1)
# model1 = Seq2Seq(enc1, dec1, SRC_PAD_IDX, device).to(device)

# attn2 = Attention(ENC_HID_DIM, DEC_HID_DIM)
# enc2 = Encoder(PIV_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# dec2 = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn2)
# model2 = Seq2Seq(enc2, dec2, PIV_PAD_IDX, device).to(device)

# models = [model1, model2]
# # fields = [EN_FIELD, DE_FIELD, FR_FIELD]
# # model = PivotSeq2Seq(models, fields, device).to(device)

# # For triangulate
# model = TriangSeq2Seq(models, OUTPUT_DIM, device).to(device)

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

model.apply(init_weights);

optimizer = optim.Adam(model.parameters(), lr=LR)

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.333)

criterion1 = nn.CrossEntropyLoss(ignore_index = DE_FIELD.vocab.stoi[DE_FIELD.pad_token])
criterion2 = nn.CrossEntropyLoss(ignore_index = FR_FIELD.vocab.stoi[FR_FIELD.pad_token])
# criterions = (criterion1, criterion2)

# for triangulate
criterions = {
    'model_0': criterion1,
    'model_1': criterion2,
}

N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')
best_train_loss = float('inf')
model_name = 'attn_enfr_160kset_3.pt'
train_log = []

for epoch in range(N_EPOCHS):
  # For 2 langs
  train_loss = trainSeq2Seq(model, train_iterator, optimizer, criterion2, CLIP)
  valid_loss = evaluateSeq2Seq(model, valid_iterator, criterion2)

  # For 3 langs
  # train_loss = trainPivot(model, train_iterator, optimizer, criterions, CLIP)
  # valid_loss = evaluatePivot(model, valid_iterator, criterions)

  scheduler.step()

  epoch_info = [model_name, scheduler.get_last_lr()[0], BATCH_SIZE, ENC_HID_DIM, 'no-num_layers', ENC_DROPOUT, DEC_DROPOUT, epoch, N_EPOCHS, train_loss, valid_loss]
  train_log.append([str(ele) for ele in epoch_info])

  # if train_loss < best_train_loss or valid_loss < best_valid_loss:
  #   best_train_loss = train_loss
  #   best_valid_loss = valid_loss
  #   torch.save({
  #       'model_state_dict': model.state_dict(),
  #       'optimizer_state_dict': optimizer.state_dict(),
  #       'scheduler_state_dict': scheduler.state_dict()
  #   }, f'/content/gdrive/MyDrive/Colab Notebooks/eaai24/{model_name}')
  #   print('SAVED MODEL')
  #   train_log = update_trainlog(train_log)

  print(f'Epoch: {epoch:02} \t Train Loss: {train_loss:.3f} \t Val. Loss: {valid_loss:.3f}')

"""## Eval"""

# FIELDS = [('src', EN_SRC), ('trg', FR_TRG)]
# start_idx = train_len + valid_len
# end_idx = start_idx + 6400
# test_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[start_idx : end_idx]))
# test_dt = Dataset(test_examples, fields=FIELDS)
# test_iterator1 = BucketIterator(
#     test_dt,
#      batch_size = BATCH_SIZE,
#      sort_within_batch = True,
#      sort_key = lambda x : len(x.src),
#      device = device)

for batch in test_iterator:
  break
batch.fields

# # For 2 langs
INPUT_DIM = 2463  #len(SRC_FIELD.vocab)
OUTPUT_DIM = 2495 #len(TRG_FIELD.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LR = 0.001
SRC_PAD_IDX = EN_FIELD.vocab.stoi[EN_FIELD.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model_infer = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
model_path = f'/content/gdrive/MyDrive/Colab Notebooks/eaai24/attn_enfr_160kset.pt'
model_infer.load_state_dict(torch.load(model_path)['model_state_dict'])

# For 3 langs
# INPUT_DIM = 2500  #len(SRC_FIELD.vocab)
# PIV_DIM = 2500  #len(PIV_FIELD.vocab)
# OUTPUT_DIM = 2500 #len(TRG_FIELD.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5
# LR = 0.001

# SRC_PAD_IDX = SRC_FIELD.vocab.stoi[SRC_FIELD.pad_token]
# PIV_PAD_IDX = PIV_FIELD.vocab.stoi[PIV_FIELD.pad_token]

# attn1 = Attention(ENC_HID_DIM, DEC_HID_DIM)
# enc1 = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# dec1 = Decoder(PIV_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn1)
# model1 = Seq2Seq(enc1, dec1, SRC_PAD_IDX, device).to(device)

# attn2 = Attention(ENC_HID_DIM, DEC_HID_DIM)
# enc2 = Encoder(PIV_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# dec2 = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn2)
# model2 = Seq2Seq(enc2, dec2, PIV_PAD_IDX, device).to(device)

# model_infer = PivotSeq2Seq(model1, model2, SRC_FIELD, PIV_FIELD, TRG_FIELD, device).to(device)
# model_path = '/content/gdrive/MyDrive/Colab Notebooks/eaai24/piv_endefr_74kset_2.pt'
# model_infer.load_state_dict(torch.load(model_path)['model_state_dict'])

# model_path = f'/content/gdrive/MyDrive/Colab Notebooks/eaai24/attn_en-fr_32k_160kset_inverse.pt'
# ckpt = torch.load(model_path)
# model.load_state_dict(ckpt['model_state_dict']) # strict=False if some dimensions are different
# optimizer.load_state_dict(ckpt['optimizer_state_dict'])
# scheduler.load_state_dict(ckpt['scheduler_state_dict'])

# test_loss = evaluate(model_infer, test_iterator, criterion2, isPivot=False, force=0)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

"""## Inference"""

def sent2tensor(src_field, trg_field, device, max_len, sentence=None):
  if sentence != None:
    if isinstance(sentence, str):
      tokens = tokenize_en(sentence)
    else:
      tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq_len, N] w/ N=1 for batch
    src_len_tensor = torch.LongTensor([len(src_indexes)]).to(device)
    return src_tensor, src_len_tensor

  trg_tensor = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]] + [0 for i in range(1, max_len)]).view(-1, 1).to(device) # [seq_len, 1]
  trg_len_tensor = torch.LongTensor([max_len]).to(device)
  return trg_tensor, trg_len_tensor

def idx2sent(trg_field, arr):
  n_sents = arr.shape[1]  # arr = [seq_len, N]
  results = []
  for i in range(n_sents):  # for each sent
    pred_sent = []
    pred = arr[:, i]
    for i in pred[1:]:  # for each word
      pred_sent.append(trg_field.vocab.itos[i])
      if i == trg_field.vocab.stoi[trg_field.eos_token]: break
    results.append(pred_sent)
  return results

def translate_sentence_seq2seq(sentence, src_field, trg_field, model: Seq2Seq, device, max_len=50):
  model.eval()
  with torch.no_grad():
    # get data
    src_tensor, src_len_tensor = sent2tensor(src_field, trg_field, device, max_len, sentence)
    trg_tensor, trg_len_tensor = sent2tensor(src_field, trg_field, device, max_len)
    data = [(src_tensor, src_len_tensor), (trg_tensor, trg_len_tensor)]
    # feed model
    output = model(data, criterion=None, teacher_forcing_ratio=0) # output = [trg_len, N, dec_emb_dim] w/ N=1
    output = output.argmax(-1).detach().cpu().numpy() # output = [seq_len, N]
    results = idx2sent(trg_field, output)
    return results

def translate_sentence_pivot(sentence, src_field, trg_field, model, device, max_len=50):  # not yet modified
  model.eval()
  with torch.no_grad():
    # get data
    src_tensor, src_len_tensor = sent2tensor(src_field, trg_field, device, max_len, sentence)
    trg_tensor, trg_len_tensor = sent2tensor(src_field, trg_field, device, max_len)
    data = [(src_tensor, src_len_tensor)] + [(trg_tensor.clone().detach().to(device), trg_len_tensor.clone().detach().to(device)) for _ in range(model.num_model)]
    # feed model
    output = model(data, criterions=None, teacher_forcing_ratio=0) # output = [trg_len, N, dec_emb_dim]
    output = output.argmax(-1).detach().cpu().numpy()
    results = idx2sent(trg_field, output)
    return results

example_idx = 432
src = vars(valid_dt.examples[example_idx])['en']
trg = vars(valid_dt.examples[example_idx])['fr']
pred = translate_sentence_seq2seq(src, EN_FIELD, FR_FIELD, model_infer, device)
print(src)
print(trg)
print(pred)

# example_idx = 432
# src = vars(valid_dt.examples[example_idx])['en']
# trg = vars(valid_dt.examples[example_idx])['fr']
# print(src)
# print(trg)
# pred = translate_sentence_pivot(src, EN_FIELD, FR_FIELD, model, device)

test_iterator_1 = BucketIterator(
     test_dt,
     batch_size = 3,
     sort_within_batch = True,
     sort_key = lambda x : len(x.fr),
     device = device)

for batch in test_iterator_1: break
batch.fields

def translate_batch_seq2seq(model_infer, iterator, trg_field, device):
  model_infer.eval()
  with torch.no_grad():
    gt_sents = []
    pred_sents = []
    for i, batch in tqdm(enumerate(iterator)):
      data = [batch.en, batch.fr] # modify based on model
      output = model_infer(data, criterion=None, teacher_forcing_ratio=0)

      pred = output.argmax(-1).detach().cpu().numpy() # [seq_len, N]
      truth = batch.fr[0].detach().cpu().numpy()  # [seq_len, N]

      gt_sents = gt_sents + idx2sent(trg_field, truth)
      pred_sents = pred_sents + idx2sent(trg_field, pred)

    return gt_sents, pred_sents

gt_sents, pred_sents = translate_batch_seq2seq(model_infer, test_iterator_1, FR_FIELD, device)

for i, (gt_sent, pred_sent) in enumerate(zip(gt_sents, pred_sents)):
  print(gt_sent)
  print(pred_sent)
  print()
  if i==5: break

"""## BLEU (not yet modified)

### Main
"""

def calculate_bleu_old(data, src_field, trg_field, model, device, max_len = 50):
  trgs = []
  pred_trgs = []
  for i, datum in tqdm(enumerate(data)):
    src = vars(datum)['src']
    trg = vars(datum)['trg']
    pred_trg = translate_sentence_seq2seq(src, src_field, trg_field, model, device, max_len)
    #cut off <eos> token
    pred_trg = pred_trg[:-1]
    pred_trgs.append(pred_trg)
    trgs.append([trg])
    if i==1500: break
  return bleu_score(pred_trgs, trgs)

def calculate_bleu(translator, data, src_field, trg_field, model, device, max_len = 50):
  trgs = []
  pred_trgs = []
  for i, datum in tqdm(enumerate(data)):
    src = vars(datum)['src']
    trg = vars(datum)['trg']
    pred = translator(src, src_field, trg_field, model, device, max_len)
    #cut off <eos> token
    pred_trgs.append(pred[:-1])
    trgs.append([trg])
    if i==2000: break
  return bleu_score(pred_trgs, trgs)

bleu_score = calculate_bleu(test_dt, EN_FIELD, FR_FIELD, model_infer, device)
print(f'BLEU score = {bleu_score*100:.2f}')

bleu_score = calculate_bleu_1piv(test_dt, EN_FIELD, DE_FIELD, FR_FIELD, model_infer, device, max_len = 50)
print(f'BLEU score = {bleu_score*100:.2f}')

"""### IF error happens: index 4 is out of bounds for dimension 0 with size 4 ==> use this"""

def get_pred_trg_corpus(data, model_infer, device, SRC_FIELD, TRG_FIELD, PIV_FIELD=None, max_len=50):
  candidate_corpus = []
  references_corpus = []
  for datum in tqdm(data):
    src = vars(datum)['src']
    trg = vars(datum)['trg']
    if isinstance(model_infer, PivotSeq2Seq):
      piv = vars(datum)['piv']
      sent1, sent = translate_sentence_1piv(src, SRC_FIELD, PIV_FIELD, TRG_FIELD, model_infer, device, max_len=max_len)
    else:
      sent, attn = translate_sentence(src, SRC_FIELD, TRG_FIELD, model_infer, device, max_len=max_len)
    candidate_corpus.append(sent[:-1])
    references_corpus.append([trg])
  return candidate_corpus, references_corpus

candidate_corpus, references_corpus = get_pred_trg_corpus(test_dt, model_infer, device, SRC_FIELD, TRG_FIELD, PIV_FIELD)

candidate_corpus[:1376], references_corpus[:1376]

i = 5000
j = 100
k = 30000
bleu_score(candidate_corpus[: i]+candidate_corpus[i+j:k], references_corpus[: i]+references_corpus[i+j:k])

candidate = ['hello', '']
references = [['hello'], ['.']]
bleu_score(candidate, references)

"""### Result

* attn_en-fr_32k.pt: BLEU = 12.65
* attn_enfr_160kset.pt: BLEU = 32.18
* piv_endefr_74kset_2.pt: BLEU = 26.33

# End
"""