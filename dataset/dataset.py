"""
@Creator: Quan Nguyen
@Date: Feb 8, 2023
@Credits: Quan Nguyen
"""

import random
import spacy
from tqdm import tqdm
from itertools import islice
from collections import defaultdict

import torchtext
import torch
from torch.utils.data import Dataset

def get_tkzer_dict(langs):
  tkz_dict = {}
  if 'en' in langs:
    spacy_en = spacy.load('en_core_web_sm')
    def tokenize_en(text):
      return [tok.text for tok in spacy_en.tokenizer(text)]
    tkz_dict['en'] = tokenize_en
  if 'de' in langs:
    spacy_de = spacy.load('de_core_news_sm')
    def tokenize_de(text):
      return [tok.text for tok in spacy_de.tokenizer(text)]
    tkz_dict['de'] = tokenize_de
  if 'fr' in langs:
    spacy_fr = spacy.load('fr_core_news_sm')
    def tokenize_fr(text):
      return [tok.text for tok in spacy_fr.tokenizer(text)]
    tkz_dict['fr'] = tokenize_fr
  if 'it' in langs:
    spacy_it = spacy.load('it_core_news_sm')
    def tokenize_it(text):
      return [tok.text for tok in spacy_it.tokenizer(text)]
    tkz_dict['it'] = tokenize_it
  if 'es' in langs:
    spacy_es = spacy.load('es_core_news_sm')
    def tokenize_es(text):
      return [tok.text for tok in spacy_es.tokenizer(text)]
    tkz_dict['es'] = tokenize_es
  if 'pt' in langs:
    spacy_pt = spacy.load('pt_core_news_sm')
    def tokenize_pt(text):
      return [tok.text for tok in spacy_pt.tokenizer(text)]
    tkz_dict['pt'] = tokenize_pt
  if 'ro' in langs:
    spacy_ro = spacy.load('ro_core_news_sm')
    def tokenize_ro(text):
      return [tok.text for tok in spacy_ro.tokenizer(text)]
    tkz_dict['ro'] = tokenize_ro
  return tkz_dict

class EuroParlDataset(Dataset):
  unk_tok = '<unk>'
  pad_tok = '<pad>'
  sos_tok = '<sos>'
  eos_tok = '<eos>'

  def __init__(self, data, langs, sort_lang, batch_size, is_shuffle_by_batch, device):
    self.langs = langs
    self.sort_lang = sort_lang
    self.batch_size = batch_size
    self.is_shuffle_by_batch = is_shuffle_by_batch
    self.device = device

    self.tkz_dict = get_tkzer_dict(self.langs)
    self.vocab_dict = self.create_vocab_dict(data)
    self.data = self.get_sort_dataset(data)

    if is_shuffle_by_batch: # to replicate bucketIterator
      self.data = self.shuffle_by_batch(self.data, self.batch_size)

  def __len__(self):
    return len(self.data)

  def get_sort_dataset(self, data):
    '''
    Return:
      each sample has 2 elements: ids and seq_len
    '''
    my_data = []
    for i, pair in enumerate(tqdm(data, desc='sort_by_seqlen')):
      tmp_dict = {}
      for lang in self.langs:
        text = pair[lang].lower()
        tokens = ['<sos>'] + self.tkz_dict[lang](text) + ['<eos>']
        tmp_dict[lang] = [self.tok2id(tokens, lang), len(tokens)]
      my_data.append(tmp_dict)
    my_data = sorted(my_data, key=lambda x : x[self.sort_lang][1], reverse=True)
    return my_data

  def shuffle_by_batch(self, data, batch_size: int):
    def chunk_split(arr, batch_size):
      arr = iter(arr)
      return list(iter(lambda: tuple(islice(arr, batch_size)), ()))
    # divide into chunks
    chunks = chunk_split(data, batch_size)
    # last item wont be sorted since may not have enough elements
    chunks_first, chunks_last = chunks[:-1], list(chunks[-1])
    random.shuffle(chunks_first)
    # recombine
    final_chunks = []
    for chunk in tqdm(chunks_first, desc='shuffle_by_batch'):
      final_chunks = final_chunks + list(chunk)
    final_chunks = final_chunks + chunks_last
    return final_chunks

  def create_vocab_dict(self, data):
    data_set = [{lang: pair[lang] for lang in self.langs} for pair in data]
    sents_dict = defaultdict(list)
    with tqdm(total=len(self.langs)*len(data_set), desc='create_vocab') as pbar:
      for lang in self.langs:
        for pair in data_set:
          text = pair[lang].lower()
          tokens = self.tkz_dict[lang](text)
          sents_dict[lang].append(tokens)
          pbar.update()

    vocab_dict = {}
    for lang in self.langs:
      vocab_dict[lang] = torchtext.vocab.build_vocab_from_iterator(sents_dict[lang], min_freq=2, specials=[self.unk_tok, self.pad_tok, self.sos_tok, self.eos_tok])
      vocab_dict[lang].set_default_index(0)

    self.unk_id = vocab_dict[self.langs[0]][self.unk_tok]
    self.pad_id = vocab_dict[self.langs[0]][self.pad_tok]
    self.sos_id = vocab_dict[self.langs[0]][self.sos_tok]
    self.eos_id = vocab_dict[self.langs[0]][self.eos_tok]
    return vocab_dict

  def tok2id(self, tokens: list, lang: str):
    '''
    Return:
      tensor of ids
    '''
    return torch.tensor(self.vocab_dict[lang](tokens))

  def id2tok(self, ids: list, lang: str):
    '''
    Return:
      list of toks
    '''
    if isinstance(ids, torch.Tensor): ids = ids.detach().clone().cpu().numpy()
    return self.vocab_dict[lang].lookup_tokens(ids)

  def __getitem__(self, idx: int):
    out = self.data[idx]
    for lang in self.langs:
        out[lang][0] = out[lang][0].to(self.device)
    return out