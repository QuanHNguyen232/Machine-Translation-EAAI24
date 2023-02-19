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

import numpy as np

import spacy
from tqdm import tqdm
import sys
sys.path.append('../')
from utils import util


cfg = util.load_cfg()

class Lang:
    def __init__(self, name: str, cfg):
        self.name = name
        self.cfg = cfg
        self.tkz = spacy.load(cfg['spacy'][name])
        self.max_len = 0
        self.word2index = {"<pad>": cfg['PAD_token'], "<sos>": cfg['SOS_token'], "<eos>": cfg['EOS_token']}
        self.index2word = {cfg['PAD_token']: "<pad>", cfg['SOS_token']: "<sos>", cfg['EOS_token']: "<eos>"}
        self.word2count = {}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence: str):
      words = [tok.text for tok in self.tkz.tokenizer(sentence)]
      for word in words:
          self.addWord(word)

    def addWord(self, word: str):
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
def normalizeStr(s: str):
    s = s.lower().strip()
    return s

def readLangs(cfg, in_lang: str='en', out_lang: str='fr', datafile: str='./data/en-fr.pkl'):
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
    
    print("Reading data...")
    data = util.load_data(datafile)
    pairs = [[normalizeStr(pair[in_lang]), normalizeStr(pair[out_lang])] for pair in data]   # lower case
    input_lang = Lang(in_lang, cfg)
    output_lang = Lang(out_lang, cfg)

    return input_lang, output_lang, pairs

def filterPair(pair: tuple):
    '''
    Args:
        pair: a pair of lang
    Return:
        (boolean) if this pair in within condition
    '''
    return cfg['MIN_LENGTH'] <= len(pair[0].split(' ')) < cfg['MAX_LENGTH'] and \
           cfg['MIN_LENGTH'] <= len(pair[1].split(' ')) < cfg['MAX_LENGTH']

def filterPairs(pairs: list):
    '''
    Args:
        pairs: list of pairs
    Return:
        list of pairs after being filtered
    '''
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(cfg, in_lang: str='en', out_lang: str='fr', datafile: str='./data/en-fr.pkl'):
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
    input_lang, output_lang, pairs = readLangs(cfg, in_lang, out_lang)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in tqdm(pairs):
        input_lang.max_len = max(input_lang.max_len, len(pair[0]))
        input_lang.addSentence(pair[0])
        output_lang.max_len = max(output_lang.max_len, len(pair[1]))
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return (input_lang, output_lang, pairs)

if __name__ == '__main__':
    print('hello')