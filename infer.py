"""
@Creator: Quan Nguyen
@Date: Jan 28, 2023
@Credits: Quan Nguyen

Trainer file
"""
#%%
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data import Dataset, Example
from torchtext.data.metrics import bleu_score

# from dataset import get_dataset_dataloader
from dataset import get_tkzer_dict, get_field_dict
from models import Seq2SeqRNN, PivotSeq2Seq, TriangSeq2Seq
from models import Seq2SeqTransformer
from models import PivotSeq2SeqMultiSrc, PivotSeq2SeqMultiSrc_2, TriangSeq2SeqMultiSrc
from models import update_trainlog, init_weights, count_parameters, save_cfg, save_model, load_model
from models import train_epoch, eval_epoch
from models import translate_sentence, translate_batch, calculate_bleu_batch
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

langs = ['en', 'de', 'es', 'it', 'ro', 'pt', 'fr']

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

model_dir = '/Accounts/turing/students/s24/nguyqu03/Quan_dir/EAAI24-NMT/dir_1/Machine-Translation-EAAI24/saved/pivMultiSrc_2_en-de-fr_shared_emb_time_0'
print('model name:', os.path.basename(model_dir))
model_cfg = util.load_cfg(os.path.join(model_dir, 'cfg.json'))

# Seq2Seq
# model_cfg['seq2seq'] = cfg['seq2seq']
# model_langs = ['en', 'fr']
# model = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
# model.apply(init_weights);

# Seq2Seq_Trans
# model_cfg['seq2seq'] = cfg['seq2seq']
# model_langs = ['en', 'fr']
# model = Seq2SeqTransformer(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)

# Piv
# model_cfg['seq2seq'] = cfg['seq2seq']
# model_langs = ['en', 'fr', 'fr', 'en']
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[2], out_lang=model_langs[3], src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
# model.apply(init_weights);

# Piv Multi-Src
model_cfg['seq2seq'] = cfg['seq2seq']
model_0 = Seq2SeqRNN(cfg=model_cfg, in_lang='en', out_lang='de', src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2SeqMultiSrc(cfg=model_cfg, submodel=model_0, device=device).to(device)
model = PivotSeq2SeqMultiSrc_2(cfg=model_cfg, submodel=model_0, device=device).to(device)
model.apply(init_weights);

# Tri
# model_cfg['seq2seq'] = cfg['seq2seq']
# model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang='fr', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# z_model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
# model = TriangSeq2Seq(cfg=cfg, models=[model_0, z_model], device=device).to(device)
# model.apply(init_weights);

# Multi-Src
# model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='de', src_pad_idx=PAD_ID, device=device).to(device)
# model = TriangSeq2SeqMultiSrc(cfg=cfg, models=[model_0], device=device).to(device)
# model.apply(init_weights);

print('model_cfg:', json.dumps(model_cfg, indent=2))

#%%

load_model(model, os.path.join(model_dir, 'ckpt_bestTrain.pt'))
if master_process: print('LOADED model')
if master_process: print(model_cfg)

#%% inference

src_lang = 'en'
trg_lang = 'fr'
# by sent
example_idx = 0
src = vars(train_dt.examples[example_idx])[src_lang]
trg = vars(train_dt.examples[example_idx])[trg_lang]
if isinstance(model, Seq2SeqTransformer):
  model.verbose = True
  res = model.translate([src], tkzer_dict, FIELD_DICT)
  pred = res['results']
else:
  pred = translate_sentence(src, tkzer_dict[src_lang], FIELD_DICT[src_lang], FIELD_DICT[trg_lang], model, model_cfg, device, src_lang, trg_lang)
print(src)
print(trg)
print(pred)
print(bleu_score([pred], [[trg]]))

# by batch
# pred_sents, gt_sents = translate_batch(model, model_cfg, test_iterator, tkzer_dict[src_lang], FIELD_DICT[src_lang], FIELD_DICT[trg_lang], device, src_lang, trg_lang, 64, 10)
# for i, (pred_sent, gt_sent) in enumerate(zip(pred_sents, gt_sents)):
#   print(pred_sent)
#   print(gt_sent)
#   print()
#   if i==3: break

# calc bleu
calculate_bleu_batch(model, model_cfg, test_iterator, tkzer_dict[src_lang], FIELD_DICT[src_lang], FIELD_DICT[trg_lang], device, src_lang, trg_lang)

