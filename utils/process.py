"""
@Creator: Quan Nguyen
@Date: Feb 8, 2023
@Credits: Quan Nguyen

process.py file for utils
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# import spacy
# import nltk
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import util


PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 128
MIN_LENGTH = 5

class Lang:
    def __init__(self, name):
        self.name = name
        self.max_len = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count PAD, SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

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
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1='en', lang2='fr', datafile: str='./data/en-fr.pkl'):
    print("Reading lines...")

    dataset = util.load_data(datafile)

    pairs = [[pair[lang1], pair[lang2]] for pair in dataset]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)

    return input_lang, output_lang, pairs

def filterPair(p):
  # p: a pair of lang
    return MIN_LENGTH <= len(p[0].split(' ')) < MAX_LENGTH and MIN_LENGTH <= len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1='en', lang2='fr', datafile: str='./data/en-fr.pkl'):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, datafile)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.max_len = max(input_lang.max_len, len(pair[0]))
        input_lang.addSentence(pair[0])
        output_lang.max_len = max(output_lang.max_len, len(pair[1]))
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs