"""
@Creator: Quan Nguyen
@Date: Aug 21st, 2023
@Credits: Quan Nguyen
"""

import os
import json
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import nn

def update_trainlog(model, data: list): # DONE Checking
  ''' Update training log w/ new losses
  Args:
      data (List): a list of infor for many epochs as tuple, each tuple has model.name, loss, etc.
  Return:
      None: new data is appended into train-log
  '''
  filename = os.path.join(model.save_dir, 'training_log.txt')
  mode = 'a'
  if not os.path.exists(filename):
    data.insert(0, ['lr', 'step', 'total_step', 'train_loss', 'valid_loss', 'date_time'])
    mode = 'w'
  with open(filename, mode) as f: # save
    for item in data:
      f.write(','.join(item) + '\n')
  print('update_trainlog SUCCESS')
  return []

def init_weights(m): # DONE Checking
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module): # DONE Checking
  num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'model has {num_param} params')
  return num_param

def save_cfg(model): # DONE Checking
  with open(os.path.join(model.save_dir, "cfg.json"), "w") as f:
    f.write(json.dumps(model.cfg))
  print('SAVED cfg')

def save_model(model, optimizer=None, scheduler=None): # DONE Checking
  save_data = {'model_state_dict': model.state_dict()}
  if optimizer is not None: save_data['optimizer_state_dict'] = optimizer.state_dict()
  if scheduler is not None: save_data['scheduler_state_dict'] = scheduler.state_dict()
  torch.save(save_data, os.path.join(model.save_dir, 'ckpt.pt'))
  print('SAVED MODEL')

def load_model(model, path, optimizer=None, scheduler=None): # DONE Checking
  # path: path to .pt file
  ckpt = torch.load(path)
  try:
    model.load_state_dict(ckpt['model_state_dict']) # strict=False if some dimensions are different
  except:
    try:
      new_state_dict = OrderedDict()
      for k, v in ckpt['model_state_dict'].items():
        name = k[7:] # remove "module."
        new_state_dict[name] = v
      model.load_state_dict(new_state_dict)
    except Exception as e:
      print(e, '\nCannot load model')
  if optimizer is not None:
    try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception as e: print(e, '\n\t optim state_dict IS NOT INCLUDED')
  if scheduler is not None:
    try: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    except Exception as e: print(e, '\n\t scheduler state_dict IS NOT INCLUDED')
  print('LOADED MODEL')

def train_epoch(master_process, model, iterator, optimizer, criterion, scheduler, curr_iter, isContinue):
  model.train()
  epoch_loss = 0.0
  if master_process:
    train_progress_bar = tqdm(iterator, desc=f'train [{model.modelname}]', position=0, leave=True)
  else:
    train_progress_bar = iterator
  for i, batch in enumerate(train_progress_bar):
    optimizer.zero_grad()
    # datas = self.prep_input(batch) # [batch.en, batch.fr]
    loss, _ = model(batch, criterion, 0.5)
    epoch_loss += loss.item()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg['CLIP'])
    optimizer.step()
    scheduler.step()

    # update iter count
    curr_iter += 1
    isContinue = curr_iter < model.cfg['NUM_ITERS']
    if not isContinue: break
  return epoch_loss / (i+1), curr_iter, isContinue

def eval_epoch(master_process, model, iterator, criterion):
  model.eval()
  epoch_loss = 0.0
  if master_process:
    eval_progress_bar = tqdm(iterator, desc=f'eval [{model.modelname}]', position=0, leave=True)
  else:
    eval_progress_bar = iterator
  with torch.no_grad():
    for batch in eval_progress_bar:
      # datas = self.prep_input(batch) # [batch.en, batch.fr]
      loss, _ = model(batch, criterion, 0) # turn off teacher forcing
      epoch_loss += loss.item()
    return epoch_loss / len(iterator)