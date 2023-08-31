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
import torchtext

def update_trainlog(model_cfg, data: list): # DONE Checking
  ''' Update training log w/ new losses
  Args:
      data (List): a list of infor for many epochs as tuple, each tuple has model.name, loss, etc.
  Return:
      None: new data is appended into train-log
  '''
  filename = os.path.join(model_cfg['save_dir'], 'training_log.txt')
  mode = 'a'
  if not os.path.exists(filename):
    data.insert(0, ['lr', 'step', 'total_step', 'train_loss', 'valid_loss', 'date_time'])
    mode = 'w'
  with open(filename, mode) as f: # save
    for item in data:
      f.write(','.join(item) + '\n')

def init_weights(m): # DONE Checking
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model: nn.Module): # DONE Checking
  num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return num_param

def save_cfg(model_cfg): # DONE Checking
  with open(os.path.join(model_cfg['save_dir'], "cfg.json"), "w") as f:
    f.write(json.dumps(model_cfg))

def save_model(model, model_cfg, isBestValid, optimizer=None, scheduler=None): # DONE Checking
  save_data = {'model_state_dict': model.state_dict()}
  if optimizer is not None: save_data['optimizer_state_dict'] = optimizer.state_dict()
  if scheduler is not None: save_data['scheduler_state_dict'] = scheduler.state_dict()
  ckpt_name = f'ckpt_bestValid.pt' if isBestValid else f'ckpt_bestTrain.pt'
  torch.save(save_data, os.path.join(model_cfg['save_dir'], ckpt_name))

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

def train_epoch(master_process, model, iterator, optimizer, criterion, scheduler, model_cfg, curr_iter, isContinue):
  model.train()
  modelname = model_cfg['model_id']
  old_loss = 0
  if torchtext.__version__ == '0.6.0': isToDict=True
  epoch_loss = 0.0
  if master_process:
    train_progress_bar = tqdm(iterator, desc=f'train [{modelname}]', position=0, leave=True)
  else:
    train_progress_bar = iterator
  for i, batch in enumerate(train_progress_bar):
    optimizer.zero_grad()
    # datas = self.prep_input(batch) # [batch.en, batch.fr]
    if isToDict: batch = vars(batch)
    loss, _ = model(batch, model_cfg, criterion, 0.5)
    epoch_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg['CLIP'])
    optimizer.step()
    scheduler.step()

    # verbose
    if (curr_iter+1)%model_cfg['verbose_interval']==0:
      old_loss = epoch_loss - old_loss
      interval_loss = old_loss / model_cfg['verbose_interval']
      print(f'curr_iter: {curr_iter} \t Train Loss: {interval_loss:.3f}')

    # update iter count
    curr_iter += 1
    isContinue = curr_iter < model_cfg['NUM_ITERS']
    if not isContinue: break
  return epoch_loss / (i+1), curr_iter, isContinue

def eval_epoch(master_process, model, iterator, criterion, model_cfg):
  model.eval()
  modelname = model_cfg['model_id']
  if torchtext.__version__ == '0.6.0': isToDict=True
  epoch_loss = 0.0
  if master_process:
    eval_progress_bar = tqdm(iterator, desc=f'eval [{modelname}]', position=0, leave=True)
  else:
    eval_progress_bar = iterator
  with torch.no_grad():
    for batch in eval_progress_bar:
      # datas = self.prep_input(batch) # [batch.en, batch.fr]
      if isToDict: batch = vars(batch)
      loss, _ = model(batch, model_cfg, criterion, 0) # turn off teacher forcing
      epoch_loss += loss.item()
    return epoch_loss / len(iterator)