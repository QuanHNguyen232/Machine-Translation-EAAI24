"""
@Creator: Quan Nguyen
@Date: Jan 28, 2023
@Credits: Quan Nguyen

Trainer file
"""
#%%
import sys
sys.path.append('../')
from dataset import dataloader
from utils import util, process
from dataset.dataloader import MyDataset
import pickle
from tqdm import tqdm
import numpy as np

from models import base, baseattn
from models.baseattn import Encoder_Attn, Decoder_Attn, Seq2Seq_Attn

import torch
import torch.nn as nn
from torch import optim

#%%

cfg = util.load_cfg()
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
# input_lang, output_lang, pairs = process.prepareData(cfg)

# util.savePickle(input_lang, output_lang, pairs)
input_lang, output_lang, pairs = util.loadPickle()

#%%
dataset_obj = MyDataset(input_lang, output_lang, pairs[:cfg['dataset_len']], cfg)
dataloader1 = torch.utils.data.DataLoader(dataset_obj, batch_size=cfg['batch_size'])

#%%
input_size_enc = input_lang.n_words
input_size_dec = output_lang.n_words

#%%
encoder_attn = Encoder_Attn(input_size_enc, cfg)
decoder_attn = Decoder_Attn(input_size_dec, cfg)
model_attn = Seq2Seq_Attn(encoder_attn, decoder_attn, cfg, device)
model_attn.apply(util.init_weights);

#%%
optimizer = optim.Adam(model_attn.parameters(), lr=cfg['learning_rate'])
criterion = nn.CrossEntropyLoss(ignore_index=cfg['PAD_token'], label_smoothing=0.5)

#%%
def train_fn(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim, criterion):
  model.train()
  total_loss = 0.0
  for i, ((en_vec, fr_vec), (en, fr)) in tqdm(enumerate(dataloader)):
    en_vec = en_vec.to(device)
    fr_vec = fr_vec.to(device)

    # Forward prop
    output = model(en_vec, fr_vec)  # (seq_len, N, target_vocab_size)

    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * batch_size that we want to send in into
    # our cost function, so we need to do some reshapin. While we're at it
    # Let's also remove the start token while we're at it
    output = output[1:].reshape(-1, output.shape[2])  # shape = (trg_len * N, target_vocab_size)
    
    fr_vec = fr_vec.permute(1, 0) # (N, seq_len) --> (seq_len, N)
    target = fr_vec[1:].reshape(-1) # shape = (trg_len * batch_size)
    # output[1:]: ignore SOS_token
    
    #https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    
    total_loss += loss.item()
    
    # if i==5: break
  return total_loss/len(dataloader)

#%%
best_train_loss = float('inf')
train_log = []
model_name = 'abc.pt'

for epoch in range(10):
  train_loss = train_fn(model_attn, dataloader1, optimizer, criterion)
  print(f'EPOCH: {epoch} \t train_loss = {train_loss}')  
  
  # valid_loss = 0.0
  # epoch_info = (model_name, cfg['learning_rate'], cfg['batch_size'], cfg['hidden_size'], cfg['num_layers'],
  #               cfg['enc_dropout'], cfg['dec_dropout'], epoch, cfg['epoch'], train_loss, valid_loss)
  # train_log.append(epoch_info)
  # if train_loss < best_train_loss:
  #   best_train_loss = train_loss
  #   train_log = update_trainlog(train_log)
  #   save_model(model_attn)
  
