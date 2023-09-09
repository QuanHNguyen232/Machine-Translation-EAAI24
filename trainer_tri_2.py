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

import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data import Dataset, Example

# from dataset import get_dataset_dataloader
from dataset import get_tkzer_dict, get_field_dict
from models import Seq2SeqRNN, PivotSeq2Seq, TriangSeq2Seq
from models import Seq2SeqTransformer
from models import PivotSeq2SeqMultiSrc, PivotSeq2SeqMultiSrc_2, TriangSeq2SeqMultiSrc, TriangSeq2SeqMultiSrc_2
from models import update_trainlog, init_weights, count_parameters, save_cfg, save_model, load_model, set_model_freeze
from models import train_epoch, eval_epoch
from utils import util

torch.cuda.empty_cache()

#%% LOAD cfg and constants

cfg = util.load_cfg()
device = cfg['device']
master_process = True

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

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

#%% get TKZERs & FIELDs

langs = ['en', 'it', 'de', 'fr']

tkzer_dict = get_tkzer_dict(langs)
FIELD_DICT = get_field_dict(tkzer_dict)


#%% LOAD dataloader

data = util.load_data(cfg['data_path'])
if master_process: print(f'data size: {len(data)}', data[8])
data_set = [[pair[lang] for lang in langs] for pair in data]
FIELDS = [(lang, FIELD_DICT[lang]) for lang in langs]

train_pt = cfg['train_len']
valid_pt = train_pt + cfg['valid_len']
test_pt = valid_pt + cfg['test_len']

train_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[: train_pt]))
valid_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[train_pt : valid_pt]))
test_examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), data_set[valid_pt : test_pt]))

train_dt = Dataset(train_examples, fields=FIELDS)
valid_dt = Dataset(valid_examples, fields=FIELDS)
test_dt = Dataset(test_examples, fields=FIELDS)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dt, valid_dt, test_dt),
     batch_size = cfg['BATCH_SIZE'],
     sort_within_batch = True,
     sort_key = lambda x : len(vars(x)['en']),
     device = device)

min_freq = 2
for lang in langs:
  FIELD_DICT[lang].build_vocab(train_dt, min_freq=min_freq)
  print(f'{lang}: {len(FIELD_DICT[lang].vocab)}')

# add lang_DIM
for lang in langs:
  cfg['seq2seq'][f'{lang}_DIM'] = len(FIELD_DICT[lang].vocab)

# train_set, train_iterator = get_dataset_dataloader(data[: train_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], True)
# valid_set, valid_iterator = get_dataset_dataloader(data[train_pt:valid_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], False)
if master_process: print(len(train_iterator), len(valid_iterator), len(test_iterator))

#%% LOAD model

# Seq2Seq
# model_langs = ['en', 'fr']
# model = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
# model.apply(init_weights)

# Seq2Seq_Trans
# model_langs = ['en', 'fr']
# model = Seq2SeqTransformer(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)

# Piv
# cfg['model_id'] = 'en-it-fr_' + cfg['model_id']
# model_langs = ['en', 'fr', 'fr', 'en']
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[2], out_lang=model_langs[3], src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
# model.apply(init_weights)

# Piv Multi-Src
# cfg['model_id'] = 'en-it-fr_' + cfg['model_id']
# model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='it', src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2SeqMultiSrc(cfg=cfg, submodel=model_0, device=device).to(device)
# model = PivotSeq2SeqMultiSrc_2(cfg=cfg, submodel=model_0, device=device, is_freeze_submodels=True).to(device)
# model.apply(init_weights)
# load_model(model.submodel, '')
# model.set_submodel_freeze()

# Tri
cfg['model_id'] = 'en-de_' + cfg['model_id']
model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='it', src_pad_idx=PAD_ID, device=device).to(device)
# model_1 = PivotSeq2SeqMultiSrc_2(cfg=cfg, submodel=model_1, device=device, is_freeze_submodels=False).to(device)
model_2 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='de', src_pad_idx=PAD_ID, device=device).to(device)
model_2 = PivotSeq2SeqMultiSrc_2(cfg=cfg, submodel=model_2, device=device, is_freeze_submodels=False).to(device)
# z_model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
model = TriangSeq2Seq(cfg=cfg, models=[model_0, model_2], device=device).to(device)
model.apply(init_weights);

# Multi-Src
# cfg['model_id'] = 'en-de-it-fr_pretrain_' + cfg['model_id']
# model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='de', src_pad_idx=PAD_ID, device=device).to(device)
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='it', src_pad_idx=PAD_ID, device=device).to(device)
# model = TriangSeq2SeqMultiSrc(cfg=cfg, models=[model_0], device=device).to(device)
# model = TriangSeq2SeqMultiSrc_2(cfg=cfg, models=[model_0, model_1], device=device, is_freeze_submodels=True).to(device)
# model.apply(init_weights)
# load_model(model_0, '/Accounts/turing/students/s24/nguyqu03/Quan_dir/EAAI24-NMT/saved/RNN_en-de_time_0/ckpt_bestTrain.pt')
# load_model(model_1, '/Accounts/turing/students/s24/nguyqu03/Quan_dir/EAAI24-NMT/saved/RNN_en-it_time_0/ckpt_bestTrain.pt')
# model.set_submodel_freeze()

model_cfg = model.cfg
save_cfg(model_cfg)
if master_process: print('SAVED cfg')
if master_process: print(model_cfg)
if cfg['use_DDP']:
  model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

#%%

#%% LOAD criterion/optim/scheduler

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.Adam(model.parameters(), lr=model_cfg['LR'])
if isinstance(model, Seq2SeqTransformer):
  optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
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
  train_loss, curr_iter, isContinue = train_epoch(master_process, model, train_iterator, optimizer, criterion, scheduler, model_cfg, curr_iter, isContinue)
  valid_loss = eval_epoch(master_process, model, valid_iterator, criterion, model_cfg)

  epoch_info = [scheduler.get_last_lr()[0], curr_iter, model_cfg['NUM_ITERS'], train_loss, valid_loss, f'{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}']
  train_log.append([str(info) for info in epoch_info])

  if train_loss < best_train_loss or valid_loss < best_valid_loss:
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      save_model(model, model_cfg, isBestValid=True, optimizer=optimizer, scheduler=scheduler)
      print('SAVED MODEL best valid')
    else:
      best_train_loss = train_loss
      save_model(model, model_cfg, isBestValid=False, optimizer=optimizer, scheduler=scheduler)
      print('SAVED MODEL best train')
    if master_process:
      update_trainlog(model_cfg, train_log)
      train_log = []
      print('update_trainlog SUCCESS')

  if isinstance(model, Seq2SeqTransformer):
    src = vars(valid_dt.examples[0])[model_langs[0]]
    trg = vars(valid_dt.examples[0])[model_langs[1]]
    res = model.translate([src], tkzer_dict, FIELD_DICT)
    text, toks = res['results'], res['tokens']
    print(f'{src}\n{trg}\n{text}\n{toks}')
  
  if master_process: print(f'Epoch: {epoch:02} \t Train Loss: {train_loss:.3f} \t Val. Loss: {valid_loss:.3f}')

  if not isContinue:
    if master_process:
      update_trainlog(model_cfg, train_log)
      print('update_trainlog SUCCESS')
    break

if cfg['use_DDP']: destroy_process_group()