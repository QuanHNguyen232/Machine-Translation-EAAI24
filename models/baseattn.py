"""
@Creator: Quan Nguyen
@Date: Feb 19, 2023
@Credits: Quan Nguyen

baseattn.py file for attention models
"""

import random

import torch
from torch import nn
import torch.nn.functional as F


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
    hidden = self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)) # (N, hidden_size) -- concat on hidden_size dim
    cell = self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)) # (N, hidden_size) -- concat on hidden_size dim
    
    return encoder_states, hidden.unsqueeze(0), cell.unsqueeze(0)
    # --> correct https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb

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
    x = x.unsqueeze(0)  # (1, N)

    embedding = self.dropout(self.embedding(x)) # (1, N, embedding_size)

    seq_len = encoder_states.shape[0]
    h_reshape = hidden.repeat(seq_len, 1, 1)  # (seq_length, N, hidden_size)
    
    # torch.cat shape = (seq_length, N, hidden_size*3)
    energy = F.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2))) # (seq_length, N, 1)
    attention = F.softmax(energy, dim=0).permute(1, 2, 0) # (seq_length, N, 1) --> (N, 1, seq_len)

    encoder_states = encoder_states.permute(1, 0, 2)  # (N, seq_len, hidden_size*2)
    context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2) # (N, 1, hidden_size*2) --> (1, N, hidden_size*2)

    rnn_input = torch.cat((context_vector, embedding), dim=2) # (1, N, hidden_size*2 + 300)

    outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
    # output = (seq_len, N, hidden_size * num_directions)
    # hidden = (num_layers * num_directions, N, hidden_size)
    # cell = (num_layers * num_directions, N, hidden_size)
      # seq_len and n directions will always be 1 in the decoder, therefore:
      # output = (1, N, hidden_size)
      # hidden = (num_layers * num_directions, N, hidden_size)
      # cell = (num_layers * num_directions, N, hidden_size)


    predictions = self.fc(outputs).squeeze(0)  # (N, vocab_len)
    
    return predictions, hidden, cell

class Seq2Seq_Attn(nn.Module):
  def __init__(self, encoder: Encoder_Attn, decoder: Decoder_Attn, cfg, device):
    super(Seq2Seq_Attn, self).__init__()
    self.cfg = cfg
    self.device = device
    self.encoder = encoder.to(self.device)
    self.decoder = decoder.to(self.device)

    assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
    assert encoder.num_layers == decoder.num_layers, "Encoder and decoder must have equal number of layers!"

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (batch_size, src_len)
    target: shape = (batch_size, target_len)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    source = source.permute(1, 0)
    target = target.permute(1, 0)

    batch_size = target.shape[1]
    target_len = target.shape[0]
    # target_vocab_size = target.n_words
    target_vocab_size = self.decoder.output_size

    # tensor to store decoder outputs
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_states, hidden, cell = self.encoder(source)

    # First input to the decoder is the <sos> tokens
    x = target[0]
    # print(f'Seq2Seq\t start x.shape = {x.shape} \t expect (batch_size)')
    for t in range(1, target_len):
      # insert input token embedding, previous hidden and previous cell states
      # receive output tensor (predictions) and new hidden and cell states
      prediction, decode_hidden, decode_cell = self.decoder(x, encoder_states, hidden, cell)
      # prediction.shape = (batch_size, target_vocab_len)
      
      # place predictions in a tensor holding predictions for each token
      outputs[t] = prediction

      #decide if we are going to use teacher forcing or not
      teacher_force = random.random() < teacher_force_ratio

      # Get the best word the Decoder predicted (index in the vocabulary)
      best_guess = prediction.argmax(1) # best_guess.shape = (batch_size)

      # With probability of teacher_force_ratio we take the actual next word otherwise we take the word that the Decoder predicted it to be.
      # Teacher Forcing is used so that the model gets used to seeing similar inputs at training and testing time, if teacher forcing is 1
      # then inputs at test time might be completely different than what the network is used to
      x = target[t] if teacher_force else best_guess

    return outputs