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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
master_process = True

if cfg['use_DDP']:
  # Initialize the process group
  init_process_group(backend='nccl')
  # Get the DDP rank
  ddp_rank = int(os.environ['RANK'])
  master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
  # Get the DDP local rank
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  # Set the cuda device
  device = f'cuda:{ddp_local_rank}'

if master_process: print(device, cfg)

#%% LOAD dataloader

data = util.load_data(cfg['data_path'])

train_pt = cfg['train_len']
valid_pt = train_pt + cfg['valid_len']
test_pt = valid_pt + cfg['test_len']

train_set, train_iterator = get_dataset_dataloader(data[: train_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], True)
valid_set, valid_iterator = get_dataset_dataloader(data[train_pt:valid_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], False)
if master_process: (len(train_iterator), len(valid_iterator))

#%% LOAD model

model = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
model_cfg = model.cfg
save_cfg(model)
if master_process: print(model_cfg)
if cfg['use_DDP']:
  model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

#%% LOAD criterion/optim/scheduler

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=model_cfg['LR'])
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(ratio*model_cfg['NUM_ITERS']) for ratio in model_cfg['scheduler']['milestones']], gamma=model_cfg['scheduler']['gamma'])
if master_process: print(scheduler.get_last_lr())


#%% train loop

curr_iter = 0
isContinue = True
best_valid_loss = float('inf')
best_train_loss = float('inf')

num_epochs = model_cfg['NUM_ITERS'] // len(train_iterator) + 1
if master_process: print('num_epochs', num_epochs)

train_log = []
train_loss = valid_loss = 0

for epoch in range(num_epochs):
  train_loss, curr_iter, isContinue = train_epoch(master_process, model, train_iterator, optimizer, criterion, scheduler, curr_iter, isContinue)
  valid_loss = eval_epoch(master_process, model, valid_iterator, criterion)

  epoch_info = [scheduler.get_last_lr()[0], curr_iter, model_cfg['NUM_ITERS'], train_loss, valid_loss, f'{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}']
  train_log.append([str(info) for info in epoch_info])

  if train_loss < best_train_loss or valid_loss < best_valid_loss:
    best_train_loss = train_loss
    best_valid_loss = valid_loss

    save_model(model=model, optimizer=optimizer, scheduler=scheduler)
    if master_process: train_log = update_trainlog(model, train_log)

  if master_process: print(f'Epoch: {epoch:02} \t Train Loss: {train_loss:.3f} \t Val. Loss: {valid_loss:.3f}')

  if not isContinue:
    train_log = update_trainlog(model, train_log)
    break

if cfg['use_DDP']: destroy_process_group()