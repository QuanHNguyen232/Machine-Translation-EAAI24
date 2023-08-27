"""
@Creator: Quan Nguyen
@Date: Aug 21st, 2023
@Credits: Quan Nguyen
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import EuroParlDataset

UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

def get_dataset_dataloader(data, langs, sort_lang, batch_size, is_shuffle_by_batch, device, use_DDP, isTrainset):

  def get_dataset(data, langs, sort_lang, batch_size, is_shuffle_by_batch, device):
    dataset = EuroParlDataset(data, langs, sort_lang, batch_size, is_shuffle_by_batch, device)
    return dataset

  def collate_fn(batch: list, langs=langs, padding_value=PAD_ID, device=device):
    out = {}
    for lang in langs:
        ids = [pair[lang][0] for pair in batch]
        ids = pad_sequence(ids, batch_first=False, padding_value=padding_value)
        lens = torch.tensor([pair[lang][1] for pair in batch]).to(device)
        out[lang] = (ids, lens)
    return out
   
  dataset = get_dataset(data, langs, sort_lang, batch_size, is_shuffle_by_batch, device)
  for tok_id, set_tok_id in zip((UNK_ID, PAD_ID, SOS_ID, EOS_ID), (dataset.unk_id, dataset.pad_id, dataset.sos_id, dataset.eos_id)): assert tok_id == set_tok_id
  if use_DDP:
    if isTrainset:
      dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    else:
      dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(dataset))
  else:
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
  return dataset, dataloader
