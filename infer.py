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
from models import update_trainlog, init_weights, count_parameters, save_cfg, save_model, load_model
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

langs = ['en', 'fr']

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
     sort_key = lambda x : len(x.en),
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
if master_process: (len(train_iterator), len(valid_iterator), len(test_iterator))

#%% LOAD model
# Seq2Seq
model_langs = ['en', 'fr']
model = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
model.apply(init_weights)
# Piv
# model_langs = ['en', 'fr', 'fr', 'en']
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[0], out_lang=model_langs[1], src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang=model_langs[2], out_lang=model_langs[3], src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
# model.apply(init_weights)
# Tri
# model_0 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang='fr', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# z_model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
# model = TriangSeq2Seq(cfg=cfg, models=[model_0, z_model], device=device).to(device)

load_model(model, 'saved/RNN_test_0/ckpt.pt')
model_cfg = model.cfg
if master_process: print('LOADED model')
if master_process: print(model_cfg)

#%% support func

def sent2tensor(tokenize_en, src_field, trg_field, device, max_len, sentence=None):
  if sentence != None:
    if isinstance(sentence, str):
      tokens = tokenize_en(sentence)
    else:
      tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq_len, N] w/ N=1 for batch
    src_len_tensor = torch.LongTensor([len(src_indexes)]).to(device)
    return src_tensor, src_len_tensor

  trg_tensor = torch.LongTensor([trg_field.vocab.stoi[trg_field.init_token]] + [0 for i in range(1, max_len)]).view(-1, 1).to(device) # [seq_len, 1]
  trg_len_tensor = torch.LongTensor([max_len]).to(device)
  return trg_tensor, trg_len_tensor

def idx2sent(trg_field, arr):
  n_sents = arr.shape[1]  # arr = [seq_len, N]
  results = []
  for i in range(n_sents):  # for each sent
    pred_sent = []
    pred = arr[:, i]
    for i in pred[1:]:  # for each word
      pred_sent.append(trg_field.vocab.itos[i])
      if i == trg_field.vocab.stoi[trg_field.eos_token]: break
    results.append(pred_sent)
  return results

def translate_sentence(sentence, tokenize_en, src_field, trg_field, model, model_cfg, device, max_len=64):
    model.eval()
    with torch.no_grad():
      # get data
      src_tensor, src_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len, sentence)
      trg_tensor, trg_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len)
      data = {'en': (src_tensor, src_len_tensor)}
      for l in langs:
        if l == 'en': continue
        data[l] = (trg_tensor.detach().clone(), trg_len_tensor.detach().clone())
      # feed model
      output, _ = model(data, model_cfg, None, 0) # output = [trg_len, N, dec_emb_dim] w/ N=1
      output = output.argmax(-1).detach().cpu().numpy() # output = [seq_len, N]
      results = idx2sent(trg_field, output)[0]
      return results[:-1] # remove <eos>

def translate_batch(model, model_cfg, iterator, tokenize_en, src_field, trg_field, device, max_len=64, batch_lim=None):
  model.eval()
  if torchtext.__version__ == '0.6.0': isToDict=True
  with torch.no_grad():
    # x_sents = []
    gt_sents, pred_sents = [], []
    for idx, batch in enumerate(tqdm(iterator)):
      # get data
      if isToDict: batch = vars(batch)
      src = batch['en'] # (data, seq_len)
      trg = batch['fr']

      _, N = src[0].shape
      trg_tensor, trg_len_tensor = sent2tensor(tokenize_en, src_field, trg_field, device, max_len)
      trg_datas = torch.cat([trg_tensor for _ in range(N)], dim=1)
      trg_lens = torch.cat([trg_len_tensor for _ in range(N)], dim=0)

      data = {'en': src}
      for l in langs:
        if l == 'en': continue
        data[l] = (trg_datas.detach().clone(), trg_lens.detach().clone())

      # feed model
      output, _ = model(data, model_cfg, None, 0)
      pred = output.argmax(-1) # [seq_len, N]

      # x_sents = x_sents + idx2sent(src_field, src[0])
      pred_sents.extend(idx2sent(trg_field, pred))
      gt_sents.extend(idx2sent(trg_field, trg[0]))

      if batch_lim!=None and idx==batch_lim: break
    pred_sents = [sent[:-1] for sent in pred_sents]
    gt_sents = [sent[:-1] for sent in gt_sents]
    return pred_sents, gt_sents

def calculate_bleu_batch(model, model_cfg, iterator, tokenizer_en, src_field, trg_field, device, max_len=64):
  pred_sents, gt_sents = translate_batch(model, model_cfg, iterator, tokenizer_en, src_field, trg_field, device, max_len=64)
  pred_sents = [pred_sent for pred_sent in pred_sents]
  gt_sents = [[gt_sent] for gt_sent in gt_sents]
  score = bleu_score(pred_sents, gt_sents)
  print(f'BLEU score = {score*100:.3f}')
  return score

#%% inference

# by sent
# example_idx = 12
# src = vars(valid_dt.examples[example_idx])['en']
# trg = vars(valid_dt.examples[example_idx])['fr']
# pred = translate_sentence(src, tkzer_dict['en'], FIELD_DICT['en'], FIELD_DICT['fr'], model, model_cfg, device)
# print(src)
# print(trg)
# print(pred)

# by batch
# pred_sents, gt_sents = translate_batch(model, model_cfg, test_iterator, tkzer_dict['en'], FIELD_DICT['en'], FIELD_DICT['fr'], device, 64, 10)
# for i, (pred_sent, gt_sent) in enumerate(zip(pred_sents, gt_sents)):
#   print(pred_sent)
#   print(gt_sent)
#   print()
#   if i==3: break

# calc bleu
calculate_bleu_batch(model, model_cfg, valid_iterator, tkzer_dict['en'], FIELD_DICT['en'], FIELD_DICT['fr'], device)

