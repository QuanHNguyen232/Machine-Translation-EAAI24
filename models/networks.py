"""
@Creator: Quan Nguyen
@Date: Aug 21st, 2023
@Credits: Quan Nguyen
"""

import torch
from torch import nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
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

class AttentionRNN(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super().__init__()
    self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
    self.v = nn.Linear(dec_hid_dim, 1, bias = False)

  def forward(self, hidden, encoder_outputs, mask):
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    #mask = [batch size, src len]
    batch_size = encoder_outputs.shape[1]
    src_len = encoder_outputs.shape[0]

    #repeat decoder hidden state src_len times
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  #hidden = [batch size, src len, dec hid dim]
    encoder_outputs = encoder_outputs.permute(1, 0, 2)  #encoder_outputs = [batch size, src len, enc hid dim * 2]
    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) #energy = [batch size, src len, dec hid dim]

    attention = self.v(energy).squeeze(2) #attention = [batch size, src len]
    attention = attention.masked_fill(mask == 0, -1e10)
    return F.softmax(attention, dim = 1)

class DecoderRNN(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
    super().__init__()
    self.output_dim = output_dim
    self.attention = attention
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def get_context_vector(self, hidden, encoder_outputs, mask):
    ''' get context vector of a single token (at time t) given hidden state (at time t-1)'''
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    #mask = [batch size, src len]
    a = self.attention(hidden, encoder_outputs, mask) #a = [batch size, src len]
    a = a.unsqueeze(1)  #a = [batch size, 1, src len]

    encoder_outputs = encoder_outputs.permute(1, 0, 2)  #encoder_outputs = [batch size, src len, enc hid dim * 2]

    weighted = torch.bmm(a, encoder_outputs)  #weighted = [batch size, 1, enc hid dim * 2]
    weighted = weighted.permute(1, 0, 2)  #weighted = [1, batch size, enc hid dim * 2]
    return a.squeeze(1), weighted

  def forward(self, input, hidden, encoder_outputs, mask):
    #input = [batch size]
    #hidden = [batch size, dec hid dim]
    #encoder_outputs = [src len, batch size, enc hid dim * 2]
    #mask = [batch size, src len]
    input = input.unsqueeze(0)  #input = [1, batch size]
    embedded = self.dropout(self.embedding(input))  #embedded = [1, batch size, emb dim]

    attn, weighted = self.get_context_vector(hidden, encoder_outputs, mask)
    #weighted = [1, batch size, enc hid dim * 2]
    #attention = [batch size, src len]

    rnn_input = torch.cat((embedded, weighted), dim = 2)  #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

    output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
    #output = [seq len, batch size, dec hid dim * n directions]
    #hidden = [n layers * n directions, batch size, dec hid dim]

    #seq len, n layers and n directions will always be 1 in this decoder, therefore:
    #output = [1, batch size, dec hid dim]
    #hidden = [1, batch size, dec hid dim]
    #this also means that output == hidden
    assert (output == hidden).all()

    embedded = embedded.squeeze(0) #embedded = [batch size, emb dim]
    output = output.squeeze(0) #output = [batch size, dec hid dim]
    weighted = weighted.squeeze(0) #weighted = [batch size, enc hid dim * 2]
    hidden = hidden.squeeze(0) #hidden = [batch size, dec hid dim]

    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))  #prediction = [batch size, output dim]
    return prediction, hidden, attn