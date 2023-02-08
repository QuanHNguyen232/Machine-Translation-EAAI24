"""
@Creator: Quan Nguyen
@Date: Feb 7, 2023
@Credits: Quan Nguyen

util.py file for utils
"""

import os
import json
import pickle
from typing import List
import pandas as pd

import torch

def calc_BLEU():
    pass

def load_cfg(filename: str='./config/configuration.json'):
    with open(filename, 'r') as jsonfile:
        cfg = json.load(jsonfile)
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cfg

def save_trainlog(data: List, filename: str='./log/training_log.txt'):
    with open(filename, 'a') as f: # save
        f.write(','.join(data))
        f.write("\n")

def load_trainlog(filename: str='./log/training_log.txt'):
    return pd.read_csv(filename)

def save_data(dataset, filename: str='./data/.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(dataset['train'], f)

def load_date(filename: str='./data/.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)