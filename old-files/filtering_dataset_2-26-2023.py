# -*- coding: utf-8 -*-
"""Filtering dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18n8ApJFjgZ36uVwopZ0VJhjnZ3sUQpRF

# Setup
"""

from google.colab import drive
drive.mount('/content/gdrive')

!pip install spacy -q

!python -m spacy download fr_core_news_sm -q
!python -m spacy download en_core_web_sm -q
!python -m spacy download de_core_news_sm -q

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

# import matplotlib.pyplot as plt
# %matplotlib inline
# import matplotlib.ticker as ticker
import numpy as np

import spacy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
device

spacy_fr = spacy.load('fr_core_news_sm')
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def load_data(filename: str):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
        print('load_data SUCCESS')
        return dataset

"""# Align with 3 langs at same time"""

enfr_set = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/en-fr.pkl')

deen_set = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/de-en.pkl')
defr_set = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/de-fr.pkl')

len(enfr_set), len(deen_set), len(defr_set)

i = 500
# print(enfr_set[i])
print(deen_set[459])
print(defr_set[444])

deen_pairs = [[pair['en'], pair['de']] for pair in deen_set]
defr_pairs = [[pair['de'], pair['fr']] for pair in defr_set]

de2_dict = {pair[0]:i for i, pair in enumerate(defr_pairs)}

align_idx_list = []
for i, pair1 in tqdm(enumerate(deen_pairs)):
  en_sent, de_sent1 = pair1
  de_sent2_idx = de2_dict.get(de_sent1, -1)
  align_idx_list.append((i, de_sent2_idx))

align_idx_list_unk_rmved = [(k, v) for k, v in align_idx_list if v!=-1]
len(align_idx_list_unk_rmved), align_idx_list_unk_rmved[:10]

save_endefr = []
thres = 0.5
for k, v in tqdm(align_idx_list_unk_rmved):
  en_sent, de_sent1 = deen_pairs[k]
  de_sent2, fr_sent = defr_pairs[v]
  save_endefr.append({
      'en': en_sent,
      'de': de_sent2 if random.random() < thres else de_sent1,
      'fr': fr_sent
  })

len(save_endefr), save_endefr[-198]

with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/en-de-fr.pkl', 'wb') as f:
  pickle.dump(save_endefr, f)

"""# Load data"""

PAD_token = 0
SOS_token = 1
EOS_token = 2
# MIN_LENGTH = 2
# MAX_LENGTH = 50

class Lang:
    def __init__(self, name):
        self.name = name
        # self.tkz = spacy.load('fr_core_news_sm') if name=='fr' else spacy.load('en_core_web_sm')
        self.max_len = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count SOS and EOS

    # def addSentence(self, sentence):
    #   words = [tok.text for tok in self.tkz.tokenizer(sentence)]
    #   # for word in sentence.split(' '):
    #   for word in words:
    #       self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: # if not in dict:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: # count++ if word already in dict
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    # s = unicodeToAscii(s)
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1='en', lang2='fr', reverse=False):
    print("Reading lines...")

    dataset = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/en-fr.pkl')
    # dataset = open('/content/eng-fra.txt', encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(pair[lang1]), normalizeString(pair[lang2])] for pair in dataset]
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in dataset]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return (input_lang, output_lang, pairs)

def read3Langs(lang1='en', lang2='de', lang3='fr'):
    print("Reading lines...")

    dataset = load_data('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/en-de-fr.pkl')

    pairs = [[normalizeString(pair[lang1]), normalizeString(pair[lang2]), normalizeString(pair[lang3])] for pair in dataset]
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in dataset]
    input_lang = Lang(lang1)
    pivot_lang = Lang(lang2)
    output_lang = Lang(lang3)

    return (input_lang, pivot_lang, output_lang, pairs)

# def prepareData(lang1='en', lang2='fr'):
#   (input_lang, output_lang, pairs) = readLangs(lang1, lang2)
#   print("Read %s sentence pairs" % len(pairs))
#   # pairs = filterPairs(pairs)
#   # print("Trimmed to %s sentence pairs" % len(pairs))

#   print("Counting words...")
#   for i in range(len(pairs)):
#     in_sent, out_sent = pairs[i]

#     en_words = [tok.text for tok in spacy_en.tokenizer(in_sent)]
#     input_lang.max_len = max(input_lang.max_len, len(en_words))
#     for en_word in en_words:
#       input_lang.addWord(en_word)

#     fr_words = [tok.text for tok in spacy_fr.tokenizer(out_sent)]
#     output_lang.max_len = max(output_lang.max_len, len(fr_words))
#     for fr_word in fr_words:
#       output_lang.addWord(fr_word)

#     pairs[i] = ['#'.join(en_words), '#'.join(fr_words)]

#   print("Counted words:")
#   print(input_lang.name, input_lang.n_words)
#   print(output_lang.name, output_lang.n_words)

#   return (input_lang, output_lang, pairs)

"""## For 2 langs"""

input_lang, output_lang, pairs = readLangs()

input_lang.name, output_lang.name, pairs[2]

len(pairs)

for i in tqdm(range(len(pairs))):
    in_sent, out_sent = pairs[i]

    en_words = [tok.text for tok in spacy_en.tokenizer(in_sent)]
    input_lang.max_len = max(input_lang.max_len, len(en_words))
    for en_word in en_words:
      input_lang.addWord(en_word)

    fr_words = [tok.text for tok in spacy_fr.tokenizer(out_sent)]
    output_lang.max_len = max(output_lang.max_len, len(fr_words))
    for fr_word in fr_words:
      output_lang.addWord(fr_word)

    pairs[i] = ['#'.join(en_words), '#'.join(fr_words)]

pairs[2]

"""## For 3 langs"""

(input_lang, pivot_lang, output_lang, pairs) = read3Langs(lang1='en', lang2='de', lang3='fr')

input_lang.name, pivot_lang.name, output_lang.name, pairs[3212]

len(pairs)

for i in tqdm(range(len(pairs))):
    in_sent, piv_sent, out_sent = pairs[i]

    en_words = [tok.text for tok in spacy_en.tokenizer(in_sent)]
    input_lang.max_len = max(input_lang.max_len, len(en_words))
    for en_word in en_words:
      input_lang.addWord(en_word)

    de_words = [tok.text for tok in spacy_de.tokenizer(piv_sent)]
    pivot_lang.max_len = max(pivot_lang.max_len, len(de_words))
    for de_word in de_words:
      pivot_lang.addWord(de_word)

    fr_words = [tok.text for tok in spacy_fr.tokenizer(out_sent)]
    output_lang.max_len = max(output_lang.max_len, len(fr_words))
    for fr_word in fr_words:
      output_lang.addWord(fr_word)

    pairs[i] = ['#'.join(en_words), '#'.join(de_words), '#'.join(fr_words)]

pairs[3212]

"""# Sort freq dict

## For 2 langs
"""

len(input_lang.word2count), len(output_lang.word2count)

sorted_freq_en = {k: v for k, v in sorted(input_lang.word2count.items(), key=lambda item: item[1], reverse=True)}
sorted_freq_fr = {k: v for k, v in sorted(output_lang.word2count.items(), key=lambda item: item[1], reverse=True)}
len(sorted_freq_en), len(sorted_freq_fr)  # (103940, 140537)

for i, (k, v) in enumerate(sorted_freq_en.items()):
  print(k, v)
  if i==5: break
print()
for i, (k, v) in enumerate(sorted_freq_fr.items()):
  print(k, v)
  if i==5: break

N_MOST_FREQ = 2500

limited_sorted_freq_en = {k: v for i, (k, v) in enumerate(sorted_freq_en.items()) if i < N_MOST_FREQ}
limited_sorted_freq_fr = {k: v for i, (k, v) in enumerate(sorted_freq_fr.items()) if i < N_MOST_FREQ}
len(limited_sorted_freq_en), len(limited_sorted_freq_fr)  # (2500, 2500)

arr = [(k, v) for i, (k, v) in enumerate(limited_sorted_freq_en.items()) if i>(N_MOST_FREQ-5)]
arr

"""## For 3 langs"""

len(input_lang.word2count), len(pivot_lang.word2count), len(output_lang.word2count)

sorted_freq_en = {k: v for k, v in sorted(input_lang.word2count.items(), key=lambda item: item[1], reverse=True)}
sorted_freq_de = {k: v for k, v in sorted(pivot_lang.word2count.items(), key=lambda item: item[1], reverse=True)}
sorted_freq_fr = {k: v for k, v in sorted(output_lang.word2count.items(), key=lambda item: item[1], reverse=True)}
len(sorted_freq_en), len(sorted_freq_de), len(sorted_freq_fr)

for i, (k, v) in enumerate(sorted_freq_en.items()):
  print(k, v)
  if i==5: break
print()
for i, (k, v) in enumerate(sorted_freq_de.items()):
  print(k, v)
  if i==5: break
print()
for i, (k, v) in enumerate(sorted_freq_fr.items()):
  print(k, v)
  if i==5: break

N_MOST_FREQ = 2500

limited_sorted_freq_en = {k: v for i, (k, v) in enumerate(sorted_freq_en.items()) if i < N_MOST_FREQ}
limited_sorted_freq_de = {k: v for i, (k, v) in enumerate(sorted_freq_de.items()) if i < N_MOST_FREQ}
limited_sorted_freq_fr = {k: v for i, (k, v) in enumerate(sorted_freq_fr.items()) if i < N_MOST_FREQ}

arr = [(k, v) for i, (k, v) in enumerate(limited_sorted_freq_de.items()) if i>(N_MOST_FREQ-5)]
arr

"""# Filter pairs"""

filtered_idx_list = []

def checkValid(sent: list, dictionary):
  valid = [word in dictionary for word in sent]
  return all(valid)

"""## For 2 langs"""

for i in tqdm(range(len(pairs))):
  en_sent, fr_sent = pairs[i]
  en_sent = en_sent.split('#')
  fr_sent = fr_sent.split('#')

  if checkValid(en_sent, limited_sorted_freq_en) and checkValid(fr_sent, limited_sorted_freq_fr):
    filtered_idx_list.append(i)

len(filtered_idx_list)

# N_MOST_FREQ = 50000 ---> len(filtered_idx_list) = 1908768
# N_MOST_FREQ = 10000 ---> len(filtered_idx_list) = 923161
# N_MOST_FREQ = 5000 ---> len(filtered_idx_list) = 429012
# N_MOST_FREQ = 2500 ---> len(filtered_idx_list) = 158629

"""## For 3 langs"""

for i in tqdm(range(len(pairs))):
  en_sent, de_sent, fr_sent = pairs[i]
  en_sent = en_sent.split('#')
  de_sent = de_sent.split('#')
  fr_sent = fr_sent.split('#')

  if checkValid(en_sent, limited_sorted_freq_en) and checkValid(de_sent, limited_sorted_freq_de) and checkValid(fr_sent, limited_sorted_freq_fr):
    filtered_idx_list.append(i)

len(filtered_idx_list)

# N_MOST_FREQ = 2500 ---> len(filtered_idx_list) = 75002

"""# Save to new dataset"""

dataset_name = '/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/en-de-fr.pkl'

save_dataset = load_data(dataset_name)
len(save_dataset), save_dataset[2310]

save_pairs = []
idx_set = set(filtered_idx_list)
for i in range(len(save_dataset)):
  if i in idx_set:
    save_pairs.append(save_dataset[i])

len(save_pairs)

with open('/content/gdrive/MyDrive/Colab Notebooks/eaai24/Datasets/endefr_75kpairs_2k5-freq-words.pkl', 'wb') as f:
  pickle.dump(save_pairs, f)