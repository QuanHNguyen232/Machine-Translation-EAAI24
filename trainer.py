"""
@Creator: Quan Nguyen
@Date: Jan 28, 2023
@Credits: Quan Nguyen

Trainer file
"""
#%%
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataset_dataloader
from models import Seq2SeqRNN, PivotSeq2Seq
from models import update_trainlog, init_weights, count_parameters, save_cfg, save_model, load_model
from models import train_epoch, eval_epoch
from utils import util

torch.cuda.empty_cache()

#%% LOAD cfg and constants

langs = ['en', 'fr']
UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

cfg = util.load_cfg()
device = cfg['device']

cfg, device

#%% LOAD dataloader

data = util.load_data(cfg['data_path'])

train_pt = cfg['train_len']
valid_pt = train_pt + cfg['valid_len']
test_pt = valid_pt + cfg['test_len']

train_set, train_iterator = get_dataset_dataloader(data[: train_pt], langs, 'en', cfg['BATCH_SIZE'], True, device)
valid_set, valid_iterator = get_dataset_dataloader(data[train_pt:valid_pt], langs, 'en', cfg['BATCH_SIZE'], True, device)
len(train_iterator), len(valid_iterator)

#%% LOAD model

model = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
save_cfg(model)
model.cfg

#%% LOAD criterion/optim/scheduler

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=model.cfg['LR'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(ratio*model.cfg['NUM_ITERS']) for ratio in model.cfg['scheduler']['milestones']], gamma=model.cfg['scheduler']['gamma'])
scheduler.get_last_lr()


#%% train loop

curr_iter = 0
isContinue = True
best_valid_loss = float('inf')
best_train_loss = float('inf')

num_epochs = model.cfg['NUM_ITERS'] // len(train_iterator) + 1
print('num_epochs', num_epochs)

train_log = []
train_loss = valid_loss = 0

for epoch in range(num_epochs):
  train_loss, curr_iter, isContinue = train_epoch(model, train_iterator, optimizer, criterion, scheduler, curr_iter, isContinue)
  valid_loss = eval_epoch(model, valid_iterator, criterion)

  epoch_info = [scheduler.get_last_lr()[0], curr_iter, model.cfg['NUM_ITERS'], train_loss, valid_loss, f'{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}']
  train_log.append([str(info) for info in epoch_info])

  if train_loss < best_train_loss or valid_loss < best_valid_loss:
    best_train_loss = train_loss
    best_valid_loss = valid_loss

    save_model(model=model, optimizer=optimizer, scheduler=scheduler)
    train_log = update_trainlog(model, train_log)

  print(f'Epoch: {epoch:02} \t Train Loss: {train_loss:.3f} \t Val. Loss: {valid_loss:.3f}')

  if not isContinue:
    train_log = update_trainlog(model, train_log)
    break

  
