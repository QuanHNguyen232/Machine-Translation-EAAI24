"""
@Creator: Quan Nguyen
@Date: Feb 7, 2023
@Credits: Quan Nguyen

base.py file for models
"""

import random

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
  def __init__(self, input_size, cfg: dict): # embedding_size, hidden_size, num_layers, p
    '''
    Args:
      input_size: size of Vocabulary (input_lang)
      embedding_size: size of vec for word2vec
      hidden_size: 1024
      num_layers: 2
      p: dropout rate = 0.5
    '''
    super(Encoder, self).__init__()
    self.cfg = cfg
    self.input_size = input_size
    self.embedding_size = cfg.get('encoder_embedding_size', 300)
    self.p = cfg.get('enc_dropout', 0.5)
    self.hidden_size = cfg.get('hidden_size', 256)
    self.num_layers = cfg.get('num_layers', 2)

    self.dropout = nn.Dropout(self.p)

    self.embedding = nn.Embedding(self.input_size, self.embedding_size, padding_idx=cfg.get('PAD_token', 0)) # output can be (batch, sent_len, embedding_size)
    self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=self.p)
  
  def forward(self, x):
    '''
    Args:
      x: has shape = (seq_len, N)

    Return:
      hidden: shape = (D∗num_layers, N, hidden_size if proj_size<=0 else proj_size)
      cell: shape = (D∗num_layers, N, hidden_size)
    '''
    embedding = self.dropout(self.embedding(x))
    # embedding shape = (seq_len, batch_size, embedding_size)

    # LSTM input: shape = (seq_len, batch_size, input_size)
    outputs, (hidden, cell) = self.rnn(embedding) # outputs shape: (seq_length, N, hidden_size)
    # outputs.shape = [seq_len, batch N, hidden_size * num_directions]
    # hidden.shape = (num_layers * num_directions, N, hidden_size)
    # cell.shape = (num_layers * num_directions, N, hidden_size)
    return hidden, cell

class Decoder(nn.Module):
  def __init__(self, output_size, cfg: dict): # input_size, embedding_size, hidden_size, num_layers, p
    '''
    input_size: size of Vocabulary
    embedding_size: size of vec for word2vec
    hidden_size: same as in Encoder
    output_size: size of Eng vocab (in case of Ger -> Eng)
    num_layers:
    p: dropout rate
    '''
    super(Decoder, self).__init__()
    self.cfg = cfg
    self.output_size = output_size
    self.hidden_size = cfg.get('hidden_size', 256)
    self.num_layers = cfg.get('num_layers', 2)
    self.p = cfg.get('dec_dropout', 0.5)
    self.embedding_size = cfg.get('decoder_embedding_size', 300)

    self.dropout = nn.Dropout(self.p)
    self.embedding = nn.Embedding(self.output_size, self.embedding_size, padding_idx=cfg.get('PAD_token', 0))
    self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=self.p)
    self.fc = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x, hidden, cell):
    '''
    Args:
      x: shape = (batch_size) because we input 1 word each time
      hidden: shape = (D * num_layers, hidden_size)
      cell: current state (for next pred)
    
    Return:
      pred: shape = (batch_size, target_vocab_len)
      hidden, cell: state for next pred
    '''
    x = x.unsqueeze(0)  # shape = (1, N) = (seq_len, N) since we use a single word and not a sentence
    # print(f'Decoder\tx.shape = {x.shape} \t expect (1, batch_size)')
    
    embedding = self.dropout(self.embedding(x)) # embedding shape = (1, N, embedding_size)
    
    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # outputs shape = (1, batch_size, hidden_size)
    # output = [seq len, N, hidden_size * num_directions]
    # hidden = [num_layers * num_directions, N, hidden_size]
    # cell = [num_layers * num_directions, N, hidden_size]

    predictions = self.fc(outputs.squeeze(0))  # predictions.shape = (N, vocab_len)
    return predictions, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
    assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (src_len, N)
    target: shape = (target_len, N)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocab_size = self.decoder.output_size

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device) # use as output prediction, init w/ zeros

    hidden, cell = self.encoder(source)

    # Grab the first input to the Decoder which will be  token
    x = target[0]
    # print(f'Seq2Seq\t start x.shape = {x.shape} \t expect (batch_size)')
    for t in range(1, target_len):
      # Use previous hidden, cell as context from encoder at start
      output, hidden, cell = self.decoder(x, hidden, cell)
      # output.shape = (batch_size, target_vocab_len)
      
      # print(f'Seq2Seq\t output.shape = {output.shape} \t expect (batch_size, target_vocab_len)')

      # Store next output prediction
      outputs[t] = output

      # Get the best word the Decoder predicted (index in the vocabulary)
      best_guess = output.argmax(1) # best_guess.shape = (N)

      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs
