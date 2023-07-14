# -*- coding: utf-8 -*-
"""final ver of direct Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/155vXNJObhYVqoU5QYJ8uI7LbFGTmMToB

Issue: Model does not converge

Reason:
  * [ ] Different dataset from tutorial [Aladdin Persson](https://www.youtube.com/watch?v=sQUqQddQtB4)
  * [X] Compare to [Pytorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), I don't have the initHidden method --> dont need (Defaults to zeros if (h_0, c_0) is not provided)
"""

from google.colab import drive
drive.mount('/content/gdrive')

!pip install spacy -q

!python -m spacy download fr_core_news_sm -q
!python -m spacy download en_core_web_sm -q
# !python -m spacy download de_core_news_sm -q

# !pip install fasttext

# Commented out IPython magic to ensure Python compatibility.
from __future__ import unicode_literals, print_function, division
import os
import io
from io import open
import unicodedata
import string
import re
import random
import pickle

import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.ticker as ticker
import numpy as np

import spacy
import nltk
from tqdm import tqdm
import gensim

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
device

cfg = {
  "PAD_token": 0,
  "SOS_token": 1,
  "EOS_token": 2,
  "MIN_LENGTH": 2,
  "MAX_LENGTH": 50,
  "max_seq_len": 64,
  "input_pad": "pre",
  "input_reverse": True,
  "spacy": {
      "en": "en_core_web_sm",
      "fr": "fr_core_news_sm",
      "de": "de_core_news_sm"
  },
  "batch_size": 128,
  "epoch": 5,
  "dataset_len": 3200,
  "encoder_embedding_size": 300,
  "decoder_embedding_size": 300,
  "hidden_size": 256,
  "num_layers": 1,
  "enc_dropout": 0.5,
  "dec_dropout": 0.5,
  "learning_rate": 0.001
}

def savePickle(input_lang, output_lang, pairs):
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/lang-en.pkl', 'wb') as f:
    pickle.dump(input_lang, f)
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/lang-fr.pkl', 'wb') as f:
    pickle.dump(output_lang, f)
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/pairs-EnFr.pkl', 'wb') as f:
    pickle.dump(pairs, f)

def loadPickle():
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/lang-en.pkl', 'rb') as f:
    input_lang = pickle.load(f)
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/lang-fr.pkl', 'rb') as f:
    output_lang = pickle.load(f)
  with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/pairs-EnFr.pkl', 'rb') as f:
    pairs = pickle.load(f)
  return input_lang, output_lang, pairs

"""# data

# Dataset
"""

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, input_lang, output_lang, pairs, cfg):
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.pairs = pairs
    self.cfg = cfg
    # self.in_tkz = spacy.load(cfg['spacy'][input_lang.name])
    # self.out_tkz = spacy.load(cfg['spacy'][output_lang.name])

  def __len__(self):
    return len(self.pairs)

  def tokenizeTxt(self, lang, text):  # Tokenizes text from a string into a list of strings (tokens)
    if lang.name == self.input_lang.name:
      return [tok.text for tok in self.input_lang.tkz.tokenizer(text)]  # self.in_tkz.tokenizer(text)
    return [tok.text for tok in self.output_lang.tkz.tokenizer(text)]  # self.out_tkz.tokenizer(text)

  def indexesFromSentence(self, lang, sentence):
    # sent2id = [lang.word2index[word] for word in sentence.split(' ')][:self.cfg['max_seq_len']]
    words = self.tokenizeTxt(lang, sentence)
    sent2id = [lang.word2index[word] for word in words]
    return [self.cfg['SOS_token']] + sent2id[:self.cfg['max_seq_len']-2] + [self.cfg['EOS_token']]  # 2 for <sos> and <eos>

  def paddingTensorFromSentence(self, lang, sentence, padding, reverse_in):
      indexes = self.indexesFromSentence(lang, sentence)
      remain_len = self.cfg['max_seq_len'] - len(indexes)
      if reverse_in:
        indexes = list(reversed(indexes))
      if padding == 'pre':
        indexes = [self.cfg['PAD_token']]*remain_len + indexes
      elif padding == 'post':
        indexes = indexes + [self.cfg['PAD_token']]*remain_len

      return torch.tensor(indexes, dtype=torch.long).view(-1) # output.shape = (cfg['max_seq_len']) = [64]

  def tensorsFromPair(self, pair):
      input_tensor = self.paddingTensorFromSentence(self.input_lang, pair[0], self.cfg['input_pad'], reverse_in=self.cfg['input_reverse'])
      target_tensor = self.paddingTensorFromSentence(self.output_lang, pair[1], 'post', reverse_in=False)

      return (input_tensor, target_tensor)  # output.shape = (cfg['max_seq_len']) = [64]

  def __getitem__(self, index):
    pair = self.pairs[index]
    # tkzed_in = [tok.text for tok in self.in_tkz.tokenizer(pair[0])]
    # tkzed_out = [tok.text for tok in self.out_tkz.tokenizer(pair[1])]
    return self.tensorsFromPair(pair), pair

"""# Testing data process

# load data
"""

class Lang:
    def __init__(self, name, cfg=cfg):
        self.name = name
        self.cfg = cfg
        self.tkz = spacy.load(cfg['spacy'][name])
        self.max_len = 0
        self.word2index = {"<pad>": cfg['PAD_token'], "<sos>": cfg['SOS_token'], "<eos>": cfg['EOS_token']}
        self.index2word = {cfg['PAD_token']: "<pad>", cfg['SOS_token']: "<sos>", cfg['EOS_token']: "<eos>"}
        self.word2count = {}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
      words = [tok.text for tok in self.tkz.tokenizer(sentence)]
      # for word in sentence.split(' '):
      for word in words:
          self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: # if not in dict:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: # count++ if word already in dict
            self.word2count[word] += 1

"""Consider using only **[lower case]** or both **[upper and lower case]**."""

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeStr(s):
    s = s.lower().strip()
    # s = unicodeToAscii(s)
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_data(filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print('load_data SUCCESS')
        return data

def readLangs(lang1='en', lang2='fr', reverse=False):
    print("Reading data...")

    data = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/dataset/en-fr.pkl')

    pairs = [[normalizeStr(pair[lang1]), normalizeStr(pair[lang2])] for pair in data]   # lower case
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return (input_lang, output_lang, pairs)

def filterPair(p):
  # p: a pair of lang
    return cfg['MIN_LENGTH'] <= len(p[0].split(' ')) < cfg['MAX_LENGTH'] and \
           cfg['MIN_LENGTH'] <= len(p[1].split(' ')) < cfg['MAX_LENGTH']

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1='en', lang2='fr'):
    (input_lang, output_lang, pairs) = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in tqdm(pairs):
        input_lang.max_len = max(input_lang.max_len, len(pair[0]))
        input_lang.addSentence(pair[0])
        output_lang.max_len = max(output_lang.max_len, len(pair[1]))
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return (input_lang, output_lang, pairs)

"""# Model

## Encoder
"""

class Encoder(nn.Module):
  def __init__(self, input_size, cfg):  # , embedding_size, hidden_size, num_layers, p
    '''
    Args:
      input_size: size of in_lang
      embedding_size: size of vec for word2vec
      hidden_size: dimensionality of the hidden and cell states = 1024
      num_layers: number of layers in the RNN = 2
      p: dropout rate = 0.5
    '''
    super(Encoder, self).__init__()
    self.cfg = cfg
    self.input_size = input_size
    self.embedding_size = cfg['encoder_embedding_size']
    self.p = cfg['enc_dropout']
    self.hidden_size = cfg['hidden_size']
    self.num_layers = cfg['num_layers']

    self.dropout = nn.Dropout(self.p)

    self.embedding = nn.Embedding(input_size, self.embedding_size, padding_idx=cfg['PAD_token']) # output can be (batch, sent_len, embedding_size)
    self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=self.p)

  def forward(self, x):
    '''
    Args:
      x: has shape = (seq_len, batch_size)
    Return:
      hidden: shape = (D∗num_layers, batch_size, hidden_size if proj_size<=0 else proj_size)
      cell: shape = (D∗num_layers, bact_size, hidden_size)
    '''
    # print(f'Encoder\t x.shape = {x.shape} \t expect (512, batch_size)')
    embedding = self.dropout(self.embedding(x))
    # embedding shape = (seq_len, batch_size, embedding_size)

    # LSTM input: shape = (seq_len, batch_size, input_size)
    outputs, (hidden, cell) = self.rnn(embedding)
    # outputs.shape = [seq_len, batch size, hidden_size * num_directions]
    # hidden.shape = (num_layers * num_directions, batch_size, hidden_size)
    # cell.shape = (num_layers * num_directions, batch_size, hidden_size)

    # note: num_directions = 1 if bidirection=False else 2
    # outputs are always from the top hidden layer
    return hidden, cell

"""## Decoder"""

class Decoder(nn.Module):
  def __init__(self, output_size, cfg):  # embedding_size, hidden_size, output_size, num_layers, p |output_dim, emb_dim, hid_dim, n_layers, dropout
    '''
    embedding_size: size of vec for word2vec
    hidden_size: same as in Encoder
    output_size: size of out_lang
    num_layers:
    p: dropout rate
    '''
    super(Decoder, self).__init__()
    self.cfg = cfg
    self.hidden_size = cfg['hidden_size']
    self.num_layers = cfg['num_layers']
    self.p = cfg['dec_dropout']
    self.output_size = output_size
    self.embedding_size = cfg['decoder_embedding_size']

    self.dropout = nn.Dropout(self.p)
    self.embedding = nn.Embedding(output_size, self.embedding_size, padding_idx=cfg['PAD_token'])
    self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=self.p)
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x, hidden, cell):
    '''
    Args:
      x: shape = (batch_size) because we input 1 word each time
      hidden: shape = (num_directions * num_layers, hidden_size)
      cell: current state (for next pred)

      #hidden = [n layers * n directions, batch size, hid dim]
      #cell = [n layers * n directions, batch size, hid dim]
    Return:
      pred: shape = (batch_size, target_vocab_len)
      hidden, cell: state for next pred
    '''
    x = x.unsqueeze(0)
    # x.shape = (1, batch_size) = (seq_len, batch_size) since we use a single word and not a sentence

    embedding = self.dropout(self.embedding(x))
    # embedding.shape = (1, batch_size, embedding_size)

    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # outputs shape = (1, batch_size, )
    # output = [seq len, batch_size, hidden_size * num_directions]
    # hidden = [num_layers * num_directions, batch_size, hidden_size]
    # cell = [num_layers * num_directions, batch_size, hidden_size]

    # seq len and n directions will always be 1 in the decoder, therefore:
    # output = [1, batch_size, hidden_size]
    # hidden = [num_layers, batch_size, hidden_size]
    # cell = [num_layers, batch_size, hidden_size]

    predictions = self.fc(outputs.squeeze(0))  # predictions.shape = (1, batch_size, vocab_len)
    # prediction = [batch size, output_size] or (batch_size, target_vocab_len)

    # predictions = predictions.squeeze(0)  # predictions.shape = (batch_size, target_vocab_len) to send to loss func

    return predictions, hidden, cell

"""## Seq2Seq"""

class Seq2Seq(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
    assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (src_len, batch_size)
    target: shape = (target_len, batch_size)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    batch_size = target.shape[1]
    target_len = target.shape[0]
    # target_vocab_size = target.n_words
    target_vocab_size = self.decoder.output_size

    # tensor to store decoder outputs
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden, cell = self.encoder(source)

    # First input to the decoder is the <sos> tokens
    x = target[0]
    # print(f'Seq2Seq\t start x.shape = {x.shape} \t expect (batch_size)')
    for t in range(1, target_len):
      # insert input token embedding, previous hidden and previous cell states
      # receive output tensor (predictions) and new hidden and cell states
      output, hidden, cell = self.decoder(x, hidden, cell)
      # output.shape = (batch_size, target_vocab_len)

      # print(f'Seq2Seq\t output.shape = {output.shape} \t expect (batch_size, target_vocab_len)')

      # place predictions in a tensor holding predictions for each token
      outputs[t] = output

      #decide if we are going to use teacher forcing or not
      teacher_force = random.random() < teacher_force_ratio

      # Get the best word the Decoder predicted (index in the vocabulary)
      best_guess = output.argmax(1) # best_guess.shape = (batch_size)
      # print(f'Seq2Seq\t best_guess.shape = {best_guess.shape} \t expect (batch_size)')

      # With probability of teacher_force_ratio we take the actual next word
      # otherwise we take the word that the Decoder predicted it to be.
      # Teacher Forcing is used so that the model gets used to seeing
      # similar inputs at training and testing time, if teacher forcing is 1
      # then inputs at test time might be completely different than what the
      # network is used to. This was a long comment.
      x = target[t] if teacher_force else best_guess

    return outputs

"""# Model Attn

## Encoder
"""

class Encoder_Attn(nn.Module):
  def __init__(self, input_size, cfg):  # embedding_size, hidden_size, num_layers, p,
    '''
    Args:
      input_size: size of in_lang
      embedding_size: size of vec for word2vec
      hidden_size: dimensionality of the hidden and cell states = 1024
      num_layers: number of layers in the RNN = 1 --> no dropout
      p: dropout rate = 0.5
    '''
    super(Encoder_Attn, self).__init__()
    self.cfg = cfg
    self.input_size = input_size
    self.embedding_size = cfg['encoder_embedding_size']
    self.p = cfg['enc_dropout']
    self.hidden_size = cfg['hidden_size']
    self.num_layers = cfg['num_layers']

    self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=self.cfg['PAD_token']) # output can be (batch, sent_len, embedding_size)
    self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, bidirectional=True)
    self.fc_hidden = nn.Linear(self.hidden_size*2, self.hidden_size)  # *2 'cause bidirection
    self.fc_cell = nn.Linear(self.hidden_size*2, self.hidden_size)
    self.dropout = nn.Dropout(self.p)

  def forward(self, x):
    '''
    Args:
      x: has shape = (batch_size, seq_len)
    Return:
      hidden: shape = (batch_size, hidden_size)
      cell: shape = (batch_size, hidden_size)
    '''
    embedding = self.dropout(self.embedding(x)) # embedding shape = (seq_len, batch_size, embedding_size)

    encoder_states, (hidden, cell) = self.rnn(embedding)  # LSTM input: shape = (seq_len, batch_size, input_size)
    # encoder_states.shape = (seq_len, N, hidden_size * num_directions)
    # hidden.shape = (num_layers * num_directions, N, hidden_size)
    # cell.shape = (num_layers * num_directions, N, hidden_size)

    # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
    # hidden [-2, :, : ] is the last of the forwards RNN
    # hidden [-1, :, : ] is the last of the backwards RNN

    # hidden = self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)) # (N, hidden_size) -- concat on hidden_size dim
    # cell = self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)) # (N, hidden_size) -- concat on hidden_size dim
    hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
    cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

    # return encoder_states, hidden.unsqueeze(0), cell.unsqueeze(0)
    return encoder_states, hidden, cell

"""## Decoder"""

class Decoder_Attn(nn.Module):
  def __init__(self, output_size, cfg):  # embedding_size, hidden_size, output_size, num_layers, p | output_dim, emb_dim, hid_dim, n_layers, dropout
    '''
    embedding_size: size of vec for word2vec
    hidden_size: same as in Encoder
    output_size: size of out_lang
    num_layers: should be 1 --> no dropout
    p: dropout rate
    '''
    super(Decoder_Attn, self).__init__()
    self.cfg = cfg
    self.hidden_size = cfg['hidden_size']
    self.num_layers = cfg['num_layers']
    self.p = cfg['dec_dropout']
    self.output_size = output_size
    self.embedding_size = cfg['decoder_embedding_size']

    self.embedding = nn.Embedding(self.output_size, self.embedding_size, padding_idx=self.cfg['PAD_token'])
    self.rnn = nn.LSTM(self.hidden_size*2 + self.embedding_size, self.hidden_size, 1)  # num_layers = 1 is a must
    self.energy = nn.Linear(self.hidden_size*3, 1) # hidden_states from encoder + prev step from decoder
    self.dropout = nn.Dropout(self.p)
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x, encoder_states, hidden, cell):
    '''
    Args:
      x: shape = (batch_size) because we input 1 word each time
      encoder_states: shape = (seq_len, batch_size, hidden_size * num_directions) --> correct
      hidden: shape = (batch_size, hidden_size)
      cell: shape = (batch_size, hidden_size)

      # hidden = [n layers * n directions, batch size, hid dim]
      # cell = [n layers * n directions, batch size, hid dim]
    Return:
      pred: shape = (batch_size, target_vocab_len)
      hidden, cell: state for next pred
    '''
    # From VIDEO
    # x = x.unsqueeze(0)  # (1, N)
    # embedding = self.dropout(self.embedding(x)) # (1, N, embedding_size)
    # seq_len = encoder_states.shape[0]
    # h_reshape = hidden.repeat(seq_len, 1, 1)  # (seq_length, N, hidden_size)

    # # torch.cat shape = (seq_length, N, hidden_size*3)
    # energy = F.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2))) # (seq_length, N, 1)
    # attention = F.softmax(energy, dim=0).permute(1, 2, 0) # (seq_length, N, 1) --> (N, 1, seq_len)
    # encoder_states = encoder_states.permute(1, 0, 2)  # (N, seq_len, hidden_size*2)
    # context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2) # (N, 1, hidden_size*2) --> (1, N, hidden_size*2)
    # rnn_input = torch.cat((context_vector, embedding), dim=2) # (1, N, hidden_size*2 + 300)

    # outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
    # # output = (seq_len, N, hidden_size * num_directions)
    # # hidden = (num_layers * num_directions, N, hidden_size)
    # # cell = (num_layers * num_directions, N, hidden_size)
    #   # seq_len and n directions will always be 1 in the decoder, therefore:
    #   # output = (1, N, hidden_size)
    #   # hidden = (num_layers * num_directions, N, hidden_size)
    #   # cell = (num_layers * num_directions, N, hidden_size)
    # predictions = self.fc(outputs).squeeze(0)  # (N, vocab_len)

    # From GITHUB https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py
    x = x.unsqueeze(0)
    # x: (1, N) where N is the batch size

    embedding = self.dropout(self.embedding(x)) # (1, N, embedding_size)
    sequence_length = encoder_states.shape[0]
    h_reshaped = hidden.repeat(sequence_length, 1, 1) # (seq_length, N, hidden_size*2)

    energy = F.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2))) # (seq_length, N, 1)

    attention = F.softmax(energy, dim=0)  # (seq_length, N, 1)

    # attention: (seq_length, N, 1), snk
    # encoder_states: (seq_length, N, hidden_size*2), snl
    # we want context_vector: (1, N, hidden_size*2), i.e knl
    context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

    rnn_input = torch.cat((context_vector, embedding), dim=2) # (1, N, hidden_size*2 + embedding_size)

    outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) # outputs shape: (1, N, hidden_size)

    predictions = self.fc(outputs).squeeze(0) # (N, hidden_size)

    return predictions, hidden, cell

"""## Seq2Seq"""

class Seq2Seq_Attn(nn.Module):
  def __init__(self, encoder: Encoder_Attn, decoder: Decoder_Attn, cfg, device):
    super(Seq2Seq_Attn, self).__init__()
    self.cfg = cfg
    self.device = device
    self.encoder = encoder.to(self.device)
    self.decoder = decoder.to(self.device)

    # assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
    # assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (batch_size, src_len)
    target: shape = (batch_size, target_len)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    source = source.permute(1, 0)
    target = target.permute(1, 0)

    batch_size = source.shape[1]
    target_len = target.shape[0]

    target_vocab_size = self.decoder.output_size  # (= target.n_words)

    # tensor to store decoder outputs
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_states, hidden, cell = self.encoder(source)

    # First input to the decoder is the <sos> tokens
    x = target[0]
    for t in range(1, target_len):
      # insert input token embedding, previous hidden and previous cell states
      # receive output tensor (predictions) and new hidden and cell states
      prediction, decode_hidden, decode_cell = self.decoder(x, encoder_states, hidden, cell)
      # prediction.shape = (batch_size, target_vocab_len)

      # place predictions in a tensor holding predictions for each token
      outputs[t] = prediction

      # Get the best word the Decoder predicted (index in the vocabulary)
      best_guess = prediction.argmax(1) # best_guess.shape = (batch_size)

      teacher_force = random.random() < teacher_force_ratio
      x = target[t] if teacher_force else best_guess

    return outputs

# with torch.no_grad():
#   x = 1
#   test_tensor = tensors_out[:x].view(x, -1)
#   mask = (test_tensor!=0)
#   extract_tensor = torch.masked_select(test_tensor, mask)
#   flip_tensor = torch.flip(extract_tensor, (0,))
#   new_tensor =
#   print(test_tensor)
#   print(mask)
#   print(extract_tensor)
#   print(flip_tensor)
#   # print(new_tensor)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

"""# Testing model attn"""

cfg = {
  "PAD_token": 0,
  "SOS_token": 1,
  "EOS_token": 2,
  "MIN_LENGTH": 2,
  "MAX_LENGTH": 50,
  "max_seq_len": 64,
  "input_pad": "post",
  "input_reverse": False,
  "spacy": {
      "en": "en_core_web_sm",
      "fr": "fr_core_news_sm",
      "de": "de_core_news_sm"
  },
  "batch_size": 32,
  "epoch": 5,
  "dataset_len": 3200,
  "encoder_embedding_size": 300,
  "decoder_embedding_size": 300,
  "hidden_size": 1024, # Needs to be the same for both RNN's
  "num_layers": 1,
  "enc_dropout": 0.0,
  "dec_dropout": 0.0,
  "learning_rate": .01
}

# input_lang, output_lang, pairs = prepareData(lang1='en', lang2='fr')
# savePickle(input_lang, output_lang, pairs)
input_lang, output_lang, pairs = loadPickle()

input_size_enc = input_lang.n_words
input_size_dec = output_lang.n_words

encoder_attn = Encoder_Attn(input_size_enc, cfg)
decoder_attn = Decoder_Attn(input_size_dec, cfg)
model_attn = Seq2Seq_Attn(encoder_attn, decoder_attn, cfg, device)
# model_attn.apply(init_weights);

optimizer = optim.Adam(model_attn.parameters(), lr=cfg['learning_rate'])

criterion = nn.CrossEntropyLoss(ignore_index=cfg['PAD_token'], label_smoothing=0.0)

def train_fn(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim, criterion: nn):
  model.train()
  total_loss = 0.0
  for i, ((en_vec, fr_vec), (en, fr)) in tqdm(enumerate(dataloader)):
    en_vec = en_vec.to(device)
    fr_vec = fr_vec.to(device)

    optimizer.zero_grad()

    # Forward prop
    output = model(en_vec, fr_vec)  # (seq_len, N, target_vocab_size)

    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * batch_size that we want to send in into
    # our cost function, so we need to do some reshapin. While we're at it
    output = output[1:].reshape(-1, output.shape[-1])  # shape = (trg_len * N, target_vocab_size)

    fr_vec = fr_vec.permute(1, 0) # (N, seq_len) --> (seq_len, N)
    target = fr_vec[1:].reshape(-1) # shape = (trg_len * batch_size)
    # output[1:]: ignore SOS_token

    # print('output', output.shape)
    # print(output)
    # print('target', target.shape)
    # print(target)

    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    total_loss += loss.item()

  return total_loss/len(dataloader)

def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn):
  model.eval()
  epoch_loss = 0

  with torch.no_grad():
    for i, ((en_vec, fr_vec), (en, fr)) in tqdm(enumerate(dataloader)):
      en_vec = en_vec.to(device)
      fr_vec = fr_vec.to(device)

      output = model(en_vec, fr_vec, 0)  # (seq_len, N, target_vocab_size)

      #trg = [trg len, batch size]
      #output = [trg len, batch size, output dim]

      output = output[1:].reshape(-1, output.shape[-1])
      fr_vec = fr_vec.permute(1, 0) # (N, seq_len) --> (seq_len, N)
      target = fr_vec[1:].reshape(-1) # shape = (trg_len * batch_size)
      #trg = [(trg len - 1) * batch size]
      #output = [(trg len - 1) * batch size, output dim]

      loss = criterion(output, target)
      epoch_loss += loss.item()

  return epoch_loss / len(dataloader)

def update_trainlog(data: list, filename: str='./log/training_log.txt'):
  print('update_trainlog SUCCESS')
  return []
def save_model(model: nn.Module):
  print('SAVED MODEL SUCCESS')

trainset = MyDataset(input_lang, output_lang, pairs[:cfg['dataset_len']], cfg)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'])
validset = MyDataset(input_lang, output_lang, pairs[cfg['dataset_len']:320], cfg)
validloader = torch.utils.data.DataLoader(validset, batch_size=cfg['batch_size'])

best_train_loss = float('inf')
train_log = []
model_name = 'abc.pt'

for epoch in range(10):
  train_loss = train_fn(model_attn, trainloader, optimizer, criterion)
  valid_loss = train_fn(model_attn, validloader, optimizer, criterion)
  print(f'EPOCH: {epoch} \t train_loss = {train_loss} \t valid_loss = {valid_loss}')

  # valid_loss = 0.0
  # epoch_info = (model_name, cfg['learning_rate'], cfg['batch_size'], cfg['hidden_size'], cfg['num_layers'],
  #               cfg['enc_dropout'], cfg['dec_dropout'], epoch, cfg['epoch'], train_loss, valid_loss)
  # train_log.append(epoch_info)
  # if train_loss < best_train_loss:
  #   best_train_loss = train_loss
  #   train_log = update_trainlog(train_log)
  #   save_model(model_attn)

# reverse then pre padding
# 200it [03:28,  1.04s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 0 	 train_loss = 7.335598840713501
# 200it [03:26,  1.03s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 1 	 train_loss = 6.436497392654419
# 200it [03:27,  1.04s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 2 	 train_loss = 6.407515485286712
# 200it [03:28,  1.04s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 3 	 train_loss = 6.4074364233016965
# 200it [03:26,  1.03s/it]EPOCH: 4 	 train_loss = 6.408039071559906

# reverse then post padding
# 200it [03:26,  1.03s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 0 	 train_loss = 7.323721137046814
# 200it [03:27,  1.04s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 1 	 train_loss = 6.433749363422394
# 200it [03:24,  1.02s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 2 	 train_loss = 6.409488241672516
# 200it [03:25,  1.03s/it]
# update_trainlog SUCCESS
# SAVED MODEL SUCCESS
# EPOCH: 3 	 train_loss = 6.407776315212249
# 200it [03:25,  1.03s/it]EPOCH: 4 	 train_loss = 6.408242735862732

"""# Test Model"""

cfg = {
  "PAD_token": 0,
  "SOS_token": 1,
  "EOS_token": 2,
  "MIN_LENGTH": 2,
  "MAX_LENGTH": 50,
  "max_seq_len": 64,
  "input_pad": "pre",
  "input_reverse": True,
  "spacy": {
      "en": "en_core_web_sm",
      "fr": "fr_core_news_sm",
      "de": "de_core_news_sm"
  },
  "batch_size": 16,
  "epoch": 5,
  "dataset_len": 320,
  "encoder_embedding_size": 300,
  "decoder_embedding_size": 300,
  "hidden_size": 128, # Needs to be the same for both RNN's
  "num_layers": 1,
  "enc_dropout": 0.5,
  "dec_dropout": 0.5,
  "learning_rate": 0.001
}

# input_lang, output_lang, pairs = prepareData(lang1='en', lang2='fr')
# savePickle(input_lang, output_lang, pairs)
input_lang, output_lang, pairs = loadPickle()

input_size_enc = input_lang.n_words
input_size_dec = output_lang.n_words

encoder_net = Encoder(input_size_enc, cfg).to(device)
decoder_net = Decoder(input_size_dec, cfg).to(device)

model_net = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer1 = optim.Adam(model_net.parameters(), lr=cfg['learning_rate'])
criterion1 = nn.CrossEntropyLoss(ignore_index=cfg['PAD_token'], label_smoothing=0.5)

def train_fn1(model: nn.Module, dataloader, optimizer, criterion):
  model.train()
  total_loss = 0.0
  for i, ((en_vec, fr_vec), (en, fr)) in tqdm(enumerate(dataloader)):
    en_vec, fr_vec = en_vec.permute(1, 0), fr_vec.permute(1, 0) # (N, seq_len) ---> (seq_len, N)
    en_vec = en_vec.to(device)
    fr_vec = fr_vec.to(device)

    # Forward prop
    output = model(en_vec, fr_vec)

    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * batch_size that we want to send in into
    # our cost function, so we need to do some reshapin. While we're at it
    # Let's also remove the start token while we're at it
    output = output[1:].reshape(-1, output.shape[2])  # shape = (trg_len * N, output_dim)
    target = fr_vec[1:].reshape(-1) # shape = (trg_len * N)
    # output[1:]: ignore SOS_token

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    total_loss += loss.item()

  return total_loss/len(dataloader)

dataset1 = MyDataset(input_lang, output_lang, pairs[:cfg['dataset_len']], cfg)
dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=cfg['batch_size'])

for epoch in range(10):
  train_loss = train_fn1(model_net, dataloader1, optimizer1, criterion1)
  print(f'EPOCH: {epoch+20} \t train_loss = {train_loss}')

# 20it [00:20,  1.02s/it]
# EPOCH: 0 	 train_loss = 10.77676510810852
# 20it [00:20,  1.03s/it]
# EPOCH: 1 	 train_loss = 9.547325325012206
# 20it [00:20,  1.05s/it]
# EPOCH: 2 	 train_loss = 9.472969818115235
# 20it [00:21,  1.08s/it]
# EPOCH: 3 	 train_loss = 9.449314308166503
# 20it [00:21,  1.05s/it]
# EPOCH: 4 	 train_loss = 9.429299545288085
# 20it [00:20,  1.03s/it]
# EPOCH: 5 	 train_loss = 9.412078332901
# 20it [00:20,  1.04s/it]
# EPOCH: 6 	 train_loss = 9.392915678024291
# 20it [00:20,  1.02s/it]
# EPOCH: 7 	 train_loss = 9.375585794448853
# 20it [00:20,  1.02s/it]
# EPOCH: 8 	 train_loss = 9.359159421920776
# 20it [00:20,  1.01s/it]
# EPOCH: 9 	 train_loss = 9.341047334671021
# 20it [00:21,  1.08s/it]
# EPOCH: 10 	 train_loss = 9.321704816818237
# 20it [00:20,  1.01s/it]
# EPOCH: 11 	 train_loss = 9.308528852462768
# 20it [00:20,  1.02s/it]
# EPOCH: 12 	 train_loss = 9.29764266014099
# 20it [00:20,  1.01s/it]
# EPOCH: 13 	 train_loss = 9.279660081863403
# 20it [00:20,  1.03s/it]
# EPOCH: 14 	 train_loss = 9.257467269897461
# 20it [00:20,  1.01s/it]
# EPOCH: 15 	 train_loss = 9.242575073242188
# 20it [00:20,  1.02s/it]
# EPOCH: 16 	 train_loss = 9.229005479812622
# 20it [00:20,  1.02s/it]
# EPOCH: 17 	 train_loss = 9.210863161087037
# 20it [00:20,  1.02s/it]
# EPOCH: 18 	 train_loss = 9.189864492416381
# 20it [00:20,  1.02s/it]
# EPOCH: 19 	 train_loss = 9.18324818611145
# 20it [00:21,  1.08s/it]
# EPOCH: 20 	 train_loss = 9.164757680892944
# 20it [00:20,  1.01s/it]
# EPOCH: 21 	 train_loss = 9.150789070129395
# 20it [00:20,  1.02s/it]
# EPOCH: 22 	 train_loss = 9.133882522583008
# 20it [00:20,  1.02s/it]
# EPOCH: 23 	 train_loss = 9.120397186279297
# 20it [00:20,  1.02s/it]
# EPOCH: 24 	 train_loss = 9.103279829025269
# 20it [00:20,  1.02s/it]
# EPOCH: 25 	 train_loss = 9.084403371810913
# 20it [00:20,  1.04s/it]
# EPOCH: 26 	 train_loss = 9.061267852783203

"""# END"""