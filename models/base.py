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
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    '''
    Args:
      input_size: size of Vocabulary (input_lang)
      embedding_size: size of vec for word2vec
      hidden_size: 1024
      num_layers: 2
      p: dropout rate = 0.5
    '''
    super(Encoder, self).__init__()
    self.dropout = nn.Dropout(p)
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.embedding = nn.Embedding(input_size, embedding_size) # output can be (batch, sent_len, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
  
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
    # print(f'Encoder\t embedding.shape = {embedding.shape} \t expect (512, batch_size, 300)')

    # embedding shape = (seq_len, batch_size, embedding_size)
    # LSTM input: shape = (seq_len, batch_size, input_size)
    outputs, (hidden, cell) = self.rnn(embedding) # outputs shape: (seq_length, N, hidden_size)
    # print(f'Encoder\t hidden.shape = {hidden.shape} \t expect ({self.num_layers}, batch_size, {self.hidden_size})')
    # print(f'Encoder\t cell.shape = {cell.shape} \t expect ({self.num_layers}, batch_size, {self.hidden_size})')

    return hidden, cell # error in return shape (expect 2D)

class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
    '''
    input_size: size of Vocabulary
    embedding_size: size of vec for word2vec
    hidden_size: same as in Encoder
    output_size: size of Eng vocab (in case of Ger -> Eng)
    num_layers:
    p: dropout rate
    '''
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)

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
    # print(f'Decoder\tx.shape = {x.shape} \t expect (batch_size)')
    x = x.unsqueeze(0)  # shape = (1, batch_size) = (seq_len, batch_size) since we use a single word and not a sentence
    # print(f'Decoder\tx.shape = {x.shape} \t expect (1, batch_size)')
    
    embedding = self.dropout(self.embedding(x)) # embedding shape = (1, batch_size, embedding_size)
    # print(f'Decoder\t embedding.shape = {embedding.shape} \t expect (1, batch_size, 300)')
    # print(f'Decoder\t hidden.shape = {hidden.shape} \t cell.shape = {cell.shape}')
    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # outputs shape = (1, batch_size, hidden_size)
    # print(f'Decoder\t outputs.shape = {outputs.shape} \t expect (1, batch_size, {self.hidden_size})')

    predictions = self.fc(outputs)  # predictions.shape = (1, batch_size, vocab_len)
    predictions = predictions.squeeze(0)  # predictions.shape = (batch_size, target_vocab_len) to send to loss func
    # print(f'Decoder\t predictions.shape = {predictions.shape} \t expect (batch_size, target_vocab_len)')
    return predictions, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (src_len, batch_size)
    target: shape = (target_len, batch_size)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    batch_size = source.shape[1]  # need modification
    target_len = target.shape[0]  # need modification
    target_vocab_size = output_lang.n_words  # need modification (len of target vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device) # use as output prediction, init w/ zeros

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
      best_guess = output.argmax(1) # best_guess.shape = (batch_size)
      # print(f'Seq2Seq\t best_guess.shape = {best_guess.shape} \t expect (batch_size)')

      # With probability of teacher_force_ratio we take the actual next word
      # otherwise we take the word that the Decoder predicted it to be.
      # Teacher Forcing is used so that the model gets used to seeing
      # similar inputs at training and testing time, if teacher forcing is 1
      # then inputs at test time might be completely different than what the
      # network is used to. This was a long comment.
      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs
