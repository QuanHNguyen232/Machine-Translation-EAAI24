"""
@Creator: Quan Nguyen
@Date: Feb 7, 2023
@Credits: Quan Nguyen

util.py file for utils
"""

import os
import json
import pickle
from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch

def calc_BLEU():
    pass

def load_cfg(filename: str='./config/configuration.json'):
    '''
    Args:
        filename (String): path + file_name of config file as json
    Return:
        config: as Dict
    '''
    with open(filename, 'r') as jsonfile:
        cfg = json.load(jsonfile)
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('load_cfg SUCCESS')
        return cfg

def update_trainlog(data: List[Tuple], filename: str='./log/training_log.txt'):
    '''
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

def load_trainlog(filename: str='./log/training_log.txt'):
    '''
    Args:
        filename (String): path + file_name
    Return:
        data: pd.DataFrame of training log, having model_name, loss, etc.
    '''
    print('load_trainlog SUCCESS')
    return pd.read_csv(filename)

def save_data(filename: str, dataset: List[Dict]):
    '''
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
    '''
    Args:
        filename (String): path + file_name
    Return:
        dataset: list of pairs. Each pair of language is a Dict.
    '''
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
        print('load_data SUCCESS')
        return dataset

def seq_len_EDA(dataset: List[Dict]):
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
