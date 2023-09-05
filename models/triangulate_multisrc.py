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
from .model_utils import init_weights, set_model_freeze

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

class TriangSeq2SeqMultiSrc_2(nn.Module):
  def __init__(self, cfg, models, device, is_freeze_submodels=False, verbose=False):
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('seq2seq', '')
    # self.cfg.pop('piv', '')
    self.cfg['model_id'] = self.modelname = 'triMultiSrc_2_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])
    self.out_lang = 'fr'
    self.output_dim = cfg['seq2seq']['fr_DIM']

    self.is_freeze_submodels = is_freeze_submodels
    self.verbose = verbose
    self.device = device
    self.add_submodel(models, cfg)
    if self.cfg['piv']['is_share_emb']:
      self.set_share_emb()
      
  def add_submodel(self, submodels, cfg):
    self.num_submodels = len(submodels)
    self.piv_langs = []
    assert self.num_submodels == 2, f'only 2 for en-de & en-it. self.num_submodels={self.num_submodels}'
    hid_dim = cfg['seq2seq']['HID_DIM']
    emb_dim = cfg['seq2seq']['EMB_DIM']
    dropout = cfg['seq2seq']['DROPOUT']
    for i in range(self.num_submodels):
      # Add submodel
      self.add_module(f'submodel_{i}', submodels[i]) # en -> piv
      self.cfg['piv'][f'submodel_{i}'] = submodels[i].cfg
      piv_lang = submodels[i].cfg['seq2seq']['model_lang']['out_lang']
      self.piv_langs.append(piv_lang)
      # Encoders:
      self.add_module(f'piv_{i}_enc', EncoderRNN(cfg['seq2seq'][f'{piv_lang}_DIM'], emb_dim, hid_dim, hid_dim, dropout))
      # Attns:
      self.add_module(f'piv_{i}_attn', AttentionRNN(hid_dim, hid_dim))
    
    # Encoder:
    self.en_enc = EncoderRNN(cfg['seq2seq']['en_DIM'], emb_dim, hid_dim, hid_dim, dropout)
    # Attn:
    self.en_attn = AttentionRNN(hid_dim, hid_dim)
    # Decoder:
    self.decoder = DecoderRNN(self.output_dim, emb_dim, hid_dim, hid_dim, dropout)
    # Hidden combine
    self.fc = nn.Linear(hid_dim * (self.num_submodels + 1), hid_dim) # = submodels + en
  
  def set_share_emb(self):
    for i in range(self.num_submodels):
      getattr(self, f'piv_{i}_enc').embedding = getattr(self, f'submodel_{i}').decoder.embedding
  
  def set_submodel_freeze(self):
    if self.is_freeze_submodels:
      for i in range(self.num_submodels):
        set_model_freeze(getattr(self, f'submodel_{i}'), isFreeze=True)

  def create_mask(self, src):
    mask = (src != PAD_ID).permute(1, 0)
    return mask

  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    if self.verbose: print('start forward')
    submodel_losses, submodel_outputs, materials = self.run_submodel(batch, model_cfg, criterion, teacher_forcing_ratio)
    
    # Run Encoders on PIV
    encoder_outputs, hiddens, masks = self.run_encoder(batch, submodel_outputs, teacher_forcing_ratio)
    if self.verbose: print('\tforward: get piv out/hid/mask from run_encoder')
    # Combine
    attn_models = [getattr(self, f'piv_{i}_attn') for i in range(self.num_submodels)] + [self.en_attn]
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
      if self.is_freeze_submodels: final_loss = pred_loss
      else:
        submodel_loss = self.compute_submodels_loss(submodel_losses)
        final_loss = (pred_loss + submodel_loss) / 3
        if self.verbose: print('end forward')
      return final_loss, outputs, None
    else:
      if self.verbose: print('end forward')
      return outputs, None

  def run_encoder(self, batch, submodel_outputs, teacher_forcing_ratio):
    ''' Run for "piv_0", "piv_1" and "en" lang
    '''
    if self.verbose: print('start run_encoder')
    enc_out_list, hid_list, mask_list = [], [], []
    for i, lang in enumerate(self.piv_langs + ['en']):
      if lang == 'en':
        src, src_len = batch[lang]
        if self.verbose: print('\trun_encoder: get src, src_len for en')
      else:
        src, src_len = batch[lang] if random.random() < teacher_forcing_ratio else self.process_output(submodel_outputs[i])
        if self.verbose: print('\trun_encoder: get src, src_len for', lang)

      # SORT: prep input for encoder
      sort_ids, unsort_ids = self.sort_by_sent_len(src_len)
      src, src_len = src[:, sort_ids], src_len[sort_ids]
      if self.verbose: print('\trun_encoder: sorted lang for', lang)

      enc = getattr(self, 'en_enc') if lang == 'en' else getattr(self, f'piv_{i}_enc')
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

    if self.verbose: print('end run_encoder')
    return enc_out_list, hid_list, mask_list
  
  def run_submodel(self, batch, model_cfg, criterion, teacher_forcing_ratio):
    ''' return:
    material: submodel_loss, submodel_output, material (enc_hid, enc_out, enc_mask)
    '''
    # Seq2Seq model already sort src by src_len in forward
    if self.verbose: print('start run_submodel')
    submodel_losses, submodel_outputs, materials = [], [], []
    for i in range(self.num_submodels):
      submodel = getattr(self, f'submodel_{i}')
      submodel_cfg = model_cfg['piv'][f'submodel_{i}']
      if self.verbose: print('\trun_submodel: get model id=', i, 'lang=', submodel_cfg['seq2seq']['model_lang']['out_lang'])
      output = submodel(batch, submodel_cfg, criterion, 0 if criterion==None else teacher_forcing_ratio)
      if criterion != None:
        submodel_losses.append(output[0])
        submodel_outputs.append(output[1])
      else:
        submodel_outputs.append(output[0])
      materials.append(output[-1])
    
    if self.verbose: print('end run_submodel')
    return submodel_losses, submodel_outputs, materials

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
  
class TriangSeq2SeqMultiSrc(nn.Module):
  def __init__(self, cfg, models: list, device, verbose=False):
    super().__init__()
    # output_dim = trg vocab size
    super().__init__()
    self.cfg = copy.deepcopy(cfg)
    self.cfg.pop('seq2seq', '')
    self.cfg.pop('piv', '')
    self.cfg['model_id'] = self.modelname = 'triMultiSrc_' + cfg['model_id']
    self.cfg['save_dir'] = self.save_dir = os.path.join(cfg['save_dir'], self.cfg['model_id'])

    self.is_train_backbone = cfg['tri']['is_train_backbone']
    self.out_lang = 'fr'
    self.output_dim = cfg['seq2seq']['fr_DIM']

    self.device = device
    self.verbose = verbose
    self.num_model = len(models)
    self.submodels = []
    self.piv_langs = []
    self.add_submodels(models, cfg)
    self.decoder = DecoderRNN(self.output_dim, cfg['seq2seq']['EMB_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['HID_DIM'], cfg['seq2seq']['DROPOUT'])
    self.fc = nn.Linear(cfg['seq2seq']['HID_DIM'] * (self.num_model + 1), cfg['seq2seq']['HID_DIM'])

    # os.makedirs(self.save_dir, exist_ok=True)
    # self.apply(init_weights)

  def add_submodels(self, models: list, cfg):
    for i, submodel in enumerate(models):
      # validity check
      assert isinstance(submodel, Seq2SeqRNN), f'{type(submodel)} != Seq2SeqRNN'
      # add submodel
      for param in submodel.parameters():
        param.requires_grad = self.is_train_backbone

      self.submodels.append(submodel)
      self.cfg['tri'][f'model_{i}'] = submodel.cfg
      hid_dim = cfg['seq2seq']['HID_DIM']
      emb_dim = cfg['seq2seq']['EMB_DIM']
      dropout = cfg['seq2seq']['DROPOUT']
      enc_in_lang = submodel.cfg['seq2seq']['model_lang']['out_lang']
      print('submodel in_lang', enc_in_lang)
      self.piv_langs.append(submodel.cfg['seq2seq']['model_lang']['out_lang'])
      self.add_module(f'enc_{i}', EncoderRNN(cfg['seq2seq'][f'{enc_in_lang}_DIM'], emb_dim, hid_dim, hid_dim, dropout))
      self.add_module(f'attn_{i}', AttentionRNN(hid_dim, hid_dim))
    # for ENG
    self.add_module(f'enc_en', EncoderRNN(cfg['seq2seq']['en_DIM'], emb_dim, hid_dim, hid_dim, dropout))
    self.add_module(f'attn_en', AttentionRNN(hid_dim, hid_dim))

  def create_mask(self, src):
    mask = (src != PAD_ID).permute(1, 0)
    return mask

  def forward(self, batch: dict, model_cfg, criterion=None, teacher_forcing_ratio=0.5):
    loss_list, output_list = self.run(batch, model_cfg, criterion, teacher_forcing_ratio)
    final_out = self.get_final_pred(batch, output_list, teacher_forcing_ratio)
    if criterion != None:
      total_loss = self.compute_submodels_loss(loss_list) + self.compute_final_pred_loss(final_out, batch[self.out_lang], criterion)
      total_loss /= (len(loss_list) + 1)
      if self.verbose: print('TriangSeq2SeqMultiSrc FORWARD')
      return total_loss, final_out, None
    else:
      if self.verbose: print('TriangSeq2SeqMultiSrc FORWARD')
      return final_out, None

  def run(self, batch, model_cfg, criterion, teacher_forcing_ratio):
    '''
    for En -> piv_lang models (direct models)
    '''
    loss_list, output_list = [], []
    for i in range(self.num_model):
      # GET MODEL
      submodel = self.submodels[i] # getattr(self, f'model_{i}')
      submodel_cfg = model_cfg['tri'][f'model_{i}']
      if self.verbose: print('run: get:', f'model_{i}')
      # FORWARD MODEL
      output = submodel(batch, submodel_cfg, criterion, 0 if criterion==None else teacher_forcing_ratio)

      if criterion != None:
        loss_list.append(output[0])
        output_list.append(output[1])
      else:
        output_list.append(output[0])
    if self.verbose: print('TriangSeq2SeqMultiSrc RUN')
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

  def get_encOut_hid(self, encname, src, src_len):
    # SORT: prep input for encoder
    sort_ids, unsort_ids = self.sort_by_sent_len(src_len)
    src, src_len = src[:, sort_ids], src_len[sort_ids]
    # for each output, feed through enc to get enc_output, hidden
    if self.verbose: print('get_encOut_hid get encname', encname)
    encoder_output, hidden = getattr(self, encname)(src, src_len)
    # UNSORT
    encoder_output, hidden = encoder_output[:, unsort_ids, :], hidden[unsort_ids, :]
    return encoder_output, hidden

  def get_final_pred(self, batch, output_list, teacher_forcing_ratio):
    # SIMILAR TO Seq2Seq's decoding step
    # output_list[0] shape = [seq_len, N, out_dim], each output is in a language
    en_tensor, en_tensor_len = batch['en']
    (trg, _) = batch[self.out_lang]
    trg_len, batch_size = trg.shape
    trg_vocab_size = self.decoder.output_dim

    attn_models = [getattr(self, f'attn_{i}') for i in range(self.num_model)] + [getattr(self, 'attn_en')]
    if self.verbose: print('get_final_pred: get attn_models')
    encoder_outputs, hiddens, masks = [], [], []
    for i, output in enumerate(output_list):
      if self.verbose: print('get_final_pred lang =', self.piv_langs[i])
      src, src_len = batch[self.piv_langs[i]] if random.random() < teacher_forcing_ratio else self.process_output(output)
      encoder_output, hidden = self.get_encOut_hid(f'enc_{i}', src, src_len)
      # add to list
      masks.append(self.create_mask(src))
      encoder_outputs.append(encoder_output) #encoder_output = [src len, batch size, enc hid dim * 2]
      hiddens.append(hidden) #hidden = [batch size, dec hid dim]
    # FOR ENG
    if self.verbose: print('get_final_pred lang = en')
    encoder_output, hidden = self.get_encOut_hid(f'enc_en', en_tensor, en_tensor_len)
    masks.append(self.create_mask(en_tensor))
    encoder_outputs.append(encoder_output) #encoder_output = [src len, batch size, enc hid dim * 2]
    hiddens.append(hidden) #hidden = [batch size, dec hid dim]

    # combine hidden by mean
    input_hidden = torch.tanh(self.fc(torch.cat(hiddens, dim=1))) # Multi-src Neural Translation by Zoph & Knight
    # input_hidden = torch.mean(torch.stack(hiddens, dim=0), dim=0) #input_hidden = [batch size, dec hid dim]
    if self.verbose: print('get_final_pred: Combine Hiddens')

    #tensor to store decoder outputs
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

    # prep for Decoder: (input, hidden, encoder_outputs: list, masks: list, attn_models: list)
    input = trg[0,:] #first input to the decoder is the <sos> tokens
    if self.verbose: print('get_final_pred', f'len(encoder_outputs)={len(encoder_outputs)}', f'len(masks)={len(masks)}', f'len(attn_models)={len(attn_models)}')
    for t in range(1, trg_len):
      #insert input token embedding, previous hidden state, all encoder hidden states and mask
      #receive output tensor (predictions) and new hidden state
      output, hidden, _ = self.decoder(input, input_hidden, encoder_outputs, masks, attn_models)

      #place predictions in a tensor holding predictions for each token
      outputs[t] = output

      #if teacher forcing, use actual next token as next input. Else, use predicted token
      input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1)
    if self.verbose: print('TriangSeq2SeqMultiSrc get_final_pred_')
    return outputs

  def sort_by_sent_len(self, sent_len):
    _, sort_ids = sent_len.sort(descending=True)
    unsort_ids = sort_ids.argsort()
    return sort_ids, unsort_ids

  def process_output(self, output): # from PIV
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