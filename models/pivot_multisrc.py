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

from .networks import EncoderRNN, AttentionRNN, DecoderRNN
from .seq2seq import Seq2SeqRNN
from .model_utils import init_weights

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

class PivotSeq2SeqMultiSrc(nn.Module):
  # For 1 models only
  def __init__(self, cfg, submodel, device, verbose=False):
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('seq2seq', '')
    self.cfg.pop('tri', '')
    self.cfg['model_id'] = self.modelname = 'pivMultiSrc_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])
    self.output_dim = cfg['seq2seq']['fr_DIM']
    
    self.verbose = verbose
    self.device = device
    self.add_submodel(submodel, cfg)

  def add_submodel(self, submodel, cfg):
    hid_dim = cfg['seq2seq']['HID_DIM']
    emb_dim = cfg['seq2seq']['EMB_DIM']
    dropout = cfg['seq2seq']['DROPOUT']
    self.add_module('submodel', submodel) # en -> piv
    self.cfg['piv']['submodel'] = submodel.cfg
    self.piv_lang = submodel.cfg['seq2seq']['model_lang']['out_lang']
    # Encoders:
    self.piv_enc = EncoderRNN(cfg['seq2seq'][f'{self.piv_lang}_DIM'], emb_dim, hid_dim, hid_dim, dropout)
    # Attns:
    self.piv_attn = AttentionRNN(hid_dim, hid_dim)
    self.en_attn = AttentionRNN(hid_dim, hid_dim)
    # Decoder:
    self.decoder = DecoderRNN(self.output_dim, emb_dim, hid_dim, hid_dim, dropout)
    # Hidden combine
    self.fc = nn.Linear(hid_dim * 2, hid_dim) # *2: piv and en
    
  def create_mask(self, src):
    mask = (src != PAD_ID).permute(1, 0)
    return mask
  
  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    if self.verbose: print('start forward')
    submodel_loss, submodel_output, material = self.run_submodel(batch, model_cfg, criterion, teacher_forcing_ratio)

    # Run Encoder on PIV
    piv_encoder_output, piv_hidden, piv_mask = self.run_encoder(batch, submodel_output, teacher_forcing_ratio)
    if self.verbose: print('\tforward: get piv out/hid/mask from run_encoder')
    # Run Encoder on EN
    en_encoder_output, en_hidden, en_mask = material['enc_out'], material['enc_hid'], material['mask']
    if self.verbose: print('\tforward: get en out/hid/mask from piv\'s run_encoder')

    # Combine
    encoder_outputs = [piv_encoder_output, en_encoder_output]
    masks = [piv_mask, en_mask]
    attn_models = [self.piv_attn, self.en_attn]
    hidden = torch.tanh(self.fc(torch.cat((piv_hidden, en_hidden), dim = 1)))
    if self.verbose: print('\tforward: combine hidden state')

    # validity
    src, _ = batch['en']
    assert (self.create_mask(src) == en_mask).all()

    # Predict (similar to seq2seq)
    trg, _ = batch['fr']
    batch_size = trg.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    if self.verbose: print('\tforward: start predicting')

    #first input to the decoder is the <sos> tokens
    input = trg[0,:]

    for t in range(1, trg_len):
      #insert input token embedding, previous hidden state, all encoder hidden states and mask
      #receive output tensor (predictions) and new hidden state
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, masks, attn_models)

      #place predictions in a tensor holding predictions for each token
      outputs[t] = output

      #if teacher forcing, use actual next token as next input. Else, use predicted token
      input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1)
    
    if criterion != None:
      pred_loss = self.compute_loss(outputs, trg, criterion)
      final_loss = (pred_loss + submodel_loss) / 2
      return final_loss, outputs, None
    else:
      return outputs, None

  def run_encoder(self, batch, submodel_output, teacher_forcing_ratio):
    ''' Run for "piv" lang
    '''
    if self.verbose: print('start run_encoder')
    if random.random() < teacher_forcing_ratio:
      src, src_len = batch[self.piv_lang]
    else:
      src, src_len = self.process_output(submodel_output)

    # SORT: prep input for encoder
    sort_ids, unsort_ids = self.sort_by_sent_len(src_len)
    src, src_len = src[:, sort_ids], src_len[sort_ids]
    if self.verbose: print('\trun_encoder: sorted piv lang')
    
    piv_encoder_output, piv_hidden = self.piv_enc(src, src_len)
    piv_encoder_output = piv_encoder_output[:, unsort_ids, :] #shape = [src len, batch size, enc hid dim * 2]
    piv_hidden = piv_hidden[unsort_ids, :] #shape = [batch size, dec hid dim]
    if self.verbose: print('\trun_encoder: get piv_enc_out, piv_hid')
    # create mask for src
    piv_mask = self.create_mask(src)  #mask = [batch size, src len]
    if self.verbose: print('\trun_encoder: get piv_mask')
    return piv_encoder_output, piv_hidden, piv_mask

  def run_submodel(self, batch, model_cfg, criterion, teacher_forcing_ratio):
    ''' return:
    material: submodel_loss, submodel_output, material (enc_hid, enc_out, enc_mask)
    '''
    # Seq2Seq model already sort src by src_len in forward
    if self.verbose: print('start run_submodel')
    submodel = getattr(self, 'submodel')
    submodel_cfg = model_cfg['piv']['submodel']
    output = submodel(batch, submodel_cfg, criterion, 0 if criterion==None else teacher_forcing_ratio)
    submodel_loss = 0
    if criterion != None:
      submodel_loss = output[0]
      submodel_output = output[1]
    else: submodel_output = output[0]
    material = output[-1]
    return submodel_loss, submodel_output, material

  def compute_submodels_loss(self, loss_list):
    total_loss = 0.0
    for loss in loss_list:
      total_loss += loss
    return total_loss

  def compute_loss(self, output, trg, criterion):
    #output = (trg_len, batch_size, trg_vocab_size)
    #trg = [seq_len, batch_size]
    output = output[1:].reshape(-1, output.shape[-1])  #output = [(trg len - 1) * batch size, output dim]
    trg = trg[1:].reshape(-1)  #trg = [(trg len - 1) * batch size]
    loss = criterion(output, trg)
    return loss
  
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
  
  def sort_by_sent_len(self, sent_len):
    _, sort_ids = sent_len.sort(descending=True)
    unsort_ids = sort_ids.argsort()
    return sort_ids, unsort_ids

class PivotSeq2SeqMultiSrc_2(PivotSeq2SeqMultiSrc):
  # For 1 models only
  def __init__(self, cfg, submodel, device, verbose=False):
    super().__init__(cfg, submodel, device, verbose)
    self.cfg['model_id'] = self.modelname = 'pivMultiSrc_2_' + cfg['model_id']

    self.add_submodel(submodel, cfg)
  
  def add_submodel(self, submodel, cfg):
    hid_dim = cfg['seq2seq']['HID_DIM']
    emb_dim = cfg['seq2seq']['EMB_DIM']
    dropout = cfg['seq2seq']['DROPOUT']
    self.add_module('submodel', submodel) # en -> piv
    self.cfg['piv']['submodel'] = submodel.cfg
    self.piv_lang = submodel.cfg['seq2seq']['model_lang']['out_lang']
    # Encoders:
    self.piv_enc = EncoderRNN(cfg['seq2seq'][f'{self.piv_lang}_DIM'], emb_dim, hid_dim, hid_dim, dropout)
    self.en_enc = EncoderRNN(cfg['seq2seq']['en_DIM'], emb_dim, hid_dim, hid_dim, dropout)
    # Attns:
    self.piv_attn = AttentionRNN(hid_dim, hid_dim)
    self.en_attn = AttentionRNN(hid_dim, hid_dim)
    # Decoder:
    self.decoder = DecoderRNN(self.output_dim, emb_dim, hid_dim, hid_dim, dropout)
    # Hidden combine
    self.fc = nn.Linear(hid_dim * 2, hid_dim) # *2: piv and en
  
  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    if self.verbose: print('start forward')
    submodel_loss, submodel_output, material = self.run_submodel(batch, model_cfg, criterion, teacher_forcing_ratio)

    # Run Encoder on PIV
    encoder_outputs, hiddens, masks = self.run_encoder(batch, submodel_output, teacher_forcing_ratio)
    if self.verbose: print('\tforward: get piv+en out/hid/mask from run_encoder')

    # Combine
    attn_models = [self.piv_attn, self.en_attn]
    hidden = torch.tanh(self.fc(torch.cat(hiddens, dim = 1)))
    if self.verbose: print('\tforward: combine hidden state')

    # Predict (similar to seq2seq)
    trg, _ = batch['fr']
    batch_size = trg.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    if self.verbose: print('\tforward: start predicting')

    #first input to the decoder is the <sos> tokens
    input = trg[0,:]

    for t in range(1, trg_len):
      #insert input token embedding, previous hidden state, all encoder hidden states and mask
      #receive output tensor (predictions) and new hidden state
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, masks, attn_models)

      #place predictions in a tensor holding predictions for each token
      outputs[t] = output

      #if teacher forcing, use actual next token as next input. Else, use predicted token
      input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1)
    
    if criterion != None:
      pred_loss = self.compute_loss(outputs, trg, criterion)
      final_loss = (pred_loss + submodel_loss) / 2
      return final_loss, outputs, None
    else:
      return outputs, None

  def run_encoder(self, batch, submodel_output, teacher_forcing_ratio):
    ''' Run for "piv" and "en" lang
    '''
    if self.verbose: print('start run_encoder')
    enc_out_list, hid_list, mask_list = [], [], []
    for lang in [self.piv_lang, 'en']:
      if lang == 'en':
        src, src_len = batch[lang]
        if self.verbose: print('\trun_encoder: get src, src_len for en')
      else:
        src, src_len = batch[lang] if random.random() < teacher_forcing_ratio else self.process_output(submodel_output)
        if self.verbose: print('\trun_encoder: get src, src_len for', lang)

      # SORT: prep input for encoder
      sort_ids, unsort_ids = self.sort_by_sent_len(src_len)
      src, src_len = src[:, sort_ids], src_len[sort_ids]
      if self.verbose: print('\trun_encoder: sorted lang for', lang)

      enc = getattr(self, 'en_enc') if lang == 'en' else getattr(self, 'piv_enc')
      encoder_output, hidden = enc(src, src_len)
      # UNSORT
      encoder_output = encoder_output[:, unsort_ids, :] #shape = [src len, batch size, enc hid dim * 2]
      hidden = hidden[unsort_ids, :] #shape = [batch size, dec hid dim]
      if self.verbose: print('\trun_encoder: get enc_out, hidden for', lang)

      mask = self.create_mask(src)
      if self.verbose: print('\trun_encoder: get mask for', lang)

      enc_out_list.append(encoder_output)
      hid_list.append(hidden)
      mask_list.append(mask)

    return enc_out_list, hid_list, mask_list