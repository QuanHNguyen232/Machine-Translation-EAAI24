"""
@Creator: Quan Nguyen
@Date: Jan 28, 2023
@Credits: Quan Nguyen

Trainer file
"""

import sys
sys.path.append('../')
from dataset import dataloader
from utils import util

# print(util.load_trainlog())
data = util.load_data('./data/en-fr.pkl')
print(data[3])