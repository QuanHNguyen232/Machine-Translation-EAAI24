"""
@Creator: Quan Nguyen
@Date: Feb 7, 2023
@Credits: Quan Nguyen

util.py file for utils
"""
import io
import os
import json
import pickle
from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn

def load_cfg(filename: str='./config/configuration.json'):
    ''' Load configuration
    Args:
        filename (String): path + file_name of config file as json
    Return:
        config: as Dict
    '''
    with open(filename, 'r') as jsonfile:
        cfg = json.load(jsonfile)
        print('load_cfg SUCCESS')
        return cfg

def update_trainlog(data: list, filename: str='./log/training_log.txt'):
    ''' Update training log w/ new losses
    Args:
        data (List): a list of infor for many epochs as tuple, each tuple has model_name, loss, etc.
        filename (String): path + file_name
    Return:
        None: new data is appended into train-log
    '''
    with open(filename, 'a') as f: # save
        for epoch in data:
            f.write(','.join(epoch))
            f.write("\n")
    print('update_trainlog SUCCESS')
    return []

def save_model(model: nn.Module, filename: str):
    torch.save(model.state_dict(), filename)
    print('SAVED MODEL')

def load_model(model: nn.Module, filename: str):
    model.load_state_dict(torch.load(filename))
    print('LOADED MODEL')
    return model

def load_trainlog(filename: str='./log/training_log.txt'):
    ''' Load training log as pandas dataframe
    Args:
        filename (String): path + file_name
    Return:
        data: pd.DataFrame of training log, having model_name, loss, etc.
    '''
    print('load_trainlog SUCCESS')
    return pd.read_csv(filename)

def save_data(filename: str, dataset: list):
    ''' Save data into Pickle file
    Args:
        dataset: list of pairs. Each pair of language is a Dict.
        filename (String): path + file_name
    Return:
        None: data is saved as pickle file
    '''
    with open(filename, 'wb') as f:
        pickle.dump(dataset['train'], f)
    print('save_data SUCCESS')

def load_data(filename: str):
    ''' Load data from Pickle file
    Args:
        filename: path + file_name
    Return:
        dataset: list of pairs. Each pair of language is a Dict.
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print('load_data SUCCESS')
        return data

def savePickle(input_lang, output_lang, pairs):
  with open(f'./data/lang-{input_lang.name}.pkl', 'wb') as f:
    pickle.dump(input_lang, f)
  with open(f'./data/lang-{output_lang.name}.pkl', 'wb') as f:
    pickle.dump(output_lang, f)
  with open(f'./data/pairs-{input_lang.name}-{output_lang.name}.pkl', 'wb') as f:
    pickle.dump(pairs, f)

def loadPickle():
  with open('./data/lang-en.pkl', 'rb') as f:
    input_lang = pickle.load(f)
  with open('./data/lang-fr.pkl', 'rb') as f:
    output_lang = pickle.load(f)
  with open('./data/pairs-en-fr.pkl', 'rb') as f:
    pairs = pickle.load(f)
  return input_lang, output_lang, pairs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def init_weights_08(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.08, 0.08)

def calc_BLEU():
    pass

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def seq_len_EDA(dataset: list):
    '''
    Args:
        dataset: list of pairs. Each pair of language is a Dict.
    Return:
        None: matplotlib plot
    '''
    max_len_1, max_len_2 = defaultdict(int), defaultdict(int)
    lang_1, lang_2 = dataset[0].keys()
    for i, pair in enumerate(dataset):
        sent_1, sent_2 = pair[lang_1], pair[lang_2]
        max_len_1[len(sent_1.split(' '))] += 1
        max_len_2[len(sent_2.split(' '))] += 1

    sort_1 = sorted(max_len_1.items(), key=lambda x:x[0])   # sort by seq_len (key)
    sort_2 = sorted(max_len_2.items(), key=lambda x:x[0])
    sort_1_key, sort_1_val = [key for key, _ in sort_1], [val for _, val in sort_1]
    sort_2_key, sort_2_val = [key for key, _ in sort_2], [val for _, val in sort_2]

    plt.plot(sort_1_key, sort_1_val)
    plt.plot(sort_2_key, sort_2_val)
