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
import os

from typing import List, Tuple, Dict

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np

# import spacy
# import nltk
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F

from utils import util


PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 128
MIN_LENGTH = 2

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
def unicodeToAscii(s: str):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s: str):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(in_lang: str='en', out_lang: str='fr', datafile: str='./data/en-fr.pkl'):
    '''
    Args:
        in_lang: language that is used as input
        out_lang: language that is used as output
        datafile: dataset file saved as pickle
    Return:
        input_lang: Lang object of in_lang w/ empty attributes
        output_lang: Lang object of out_lang w/ empty attributes
        pairs: list of pairs of sentences (each pair is in_lang - out_lang)
    '''
    if not os.path.exists(datafile):
        print('Cannot find', datafile)
        return None

    dataset = util.load_data(datafile)

    pairs = [[pair[in_lang], pair[out_lang]] for pair in dataset]
    input_lang = Lang(in_lang)
    output_lang = Lang(out_lang)

    return input_lang, output_lang, pairs

def filterPair(pair: Tuple):
    '''
    Args:
        pair: a pair of lang
    Return:
        (boolean) if this pair in within condition
    '''
    return (MIN_LENGTH <= len(pair[0].split(' ')) < MAX_LENGTH) and (MIN_LENGTH <= len(pair[1].split(' ')) < MAX_LENGTH)

def filterPairs(pairs: List(Tuple)):
    '''
    Args:
        pairs: list of pairs
    Return:
        list of pairs after being filtered
    '''
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(in_lang: str='en', out_lang: str='fr', datafile: str='./data/en-fr.pkl'):
    '''
    Args:
        in_lang: language that is used as input
        out_lang: language that is used as output
        datafile: dataset file saved as pickle
    Return:
        input_lang: Lang object of in_lang w/ attributes updated
        output_lang: Lang object of out_lang w/ attributes updated
        pairs: list of pairs of sentences (each pair is in_lang - out_lang)
    '''
    input_lang, output_lang, pairs = readLangs(in_lang, out_lang, datafile)
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