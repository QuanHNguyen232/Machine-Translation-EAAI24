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
from torchtext.data.metrics import bleu_score

from dataset import get_dataset_dataloader
from models import Seq2SeqRNN, PivotSeq2Seq, PivotSeq2SeqMultiSrc, TriangSeq2Seq, TriangSeq2SeqMultiSrc
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

if master_process: print(device, cfg)

#%% LOAD dataloader

data = util.load_data(cfg['data_path'])

train_pt = cfg['train_len']
valid_pt = train_pt + cfg['valid_len']
test_pt = valid_pt + cfg['test_len']

train_set, train_iterator = get_dataset_dataloader(data[: train_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], True)
valid_set, valid_iterator = get_dataset_dataloader(data[train_pt:valid_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], True)
test_set, test_iterator = get_dataset_dataloader(data[valid_pt:test_pt], langs, 'en', cfg['BATCH_SIZE'], True, device, cfg['use_DDP'], True)
if master_process: (len(test_iterator))

#%% LOAD model
# Seq2Seq
model = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# Piv
# model_1 = Seq2SeqRNN(cfg=cfg, in_lang='en', out_lang='fr', src_pad_idx=PAD_ID, device=device).to(device)
# model_2 = Seq2SeqRNN(cfg=cfg, in_lang='fr', out_lang='en', src_pad_idx=PAD_ID, device=device).to(device)
# model = PivotSeq2Seq(cfg=cfg, models=[model_1, model_2], device=device).to(device)
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

def sent2tensor(src_field, trg_field, device, max_len, sentence=None): # already in Dataset.tok2id
  if sentence != None:
    if isinstance(sentence, str):
      tokens = tokenize_en(sentence)
    else:
      tokens = [token.lower() for token in sentence]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_field[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq_len, N] w/ N=1 for batch
    src_len_tensor = torch.LongTensor([len(src_indexes)]).to(device)
    return src_tensor, src_len_tensor

  trg_tensor = torch.LongTensor([SOS_ID] + [0 for i in range(1, max_len)]).view(-1, 1).to(device) # [seq_len, 1] as batch = 1
  trg_len_tensor = torch.LongTensor([max_len]).to(device)
  return trg_tensor, trg_len_tensor

def idx2sent(trg_field, arr): # already in Dataset.id2tok
  n_sents = arr.shape[1]  # arr = [seq_len, N]
  results = []
  for i in range(n_sents):  # for each sent
    pred_sent = []
    pred = arr[:, i]
    for i in pred[1:]:  # for each word
      pred_sent.append(trg_field.lookup_tokens([i])[0])
      if i == EOS_ID: break
    results.append(pred_sent)
  return results

def translate_batch(model, model_cfg, iterator, src_field, trg_field, device, max_len=64):
  model.eval()
  langs = ['fr', 'de', 'it', 'es', 'pt', 'ro']
  with torch.no_grad():
    # x_sents = []
    gt_sents, pred_sents = [], []
    for idx, batch in enumerate(tqdm(iterator)):
      # get data
      src = batch['en'] # (data, seq_len)
      trg = batch['fr']

      _, N = src[0].shape
      trg_tensor, trg_len_tensor = sent2tensor(src_field, trg_field, device, max_len)
      trg_datas = torch.cat([trg_tensor for _ in range(N)], dim=1)
      trg_lens = torch.cat([trg_len_tensor for _ in range(N)], dim=0)
      
      data = {'en': src}
      for l in langs:
        data[l] = (trg_datas.detach().clone(), trg_lens.detach().clone())

      # feed model
      output = model(data, model_cfg, None, 0)
      pred = output.argmax(-1) # [seq_len, N]

      # x_sents = x_sents + idx2sent(src_field, src[0])
      pred_sents = pred_sents + idx2sent(trg_field, pred)
      gt_sents = gt_sents + idx2sent(trg_field, trg[0])
      
      # for pred, gt in zip(pred_sents, gt_sents):
      #   print(pred, '\n', gt, end='\n\n')
    return gt_sents, pred_sents

def calculate_bleu_batch(model, model_cfg, iterator, src_field, trg_field, device, max_len=64):
  gt_sents, pred_sents = translate_batch(model, model_cfg, iterator, src_field, trg_field, device, max_len=64)
  gt_sents = [[gt_sent[:-1]] for gt_sent in gt_sents]
  pred_sents = [pred_sent[:-1] for pred_sent in pred_sents]
  score = bleu_score(pred_sents, gt_sents)
  print(f'BLEU score = {score*100:.3f}')

#%% inference

# gt_sents, pred_sents = translate_batch(model, model_cfg, valid_iterator, train_set.vocab_dict['en'], train_set.vocab_dict['fr'], device)
# for i, (gt_sent, pred_sent) in enumerate(zip(gt_sents, pred_sents)):
#   print(gt_sent[:-1])
#   print(pred_sent[:-1])
#   print()
#   if i==3: break


calculate_bleu_batch(model, model_cfg, test_iterator, train_set.vocab_dict['en'], train_set.vocab_dict['fr'], device)


