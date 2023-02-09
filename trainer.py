"""
@Creator: Quan Nguyen
@Date: Jan 28, 2023
@Credits: Quan Nguyen

Trainer file
"""
#%%
import sys
sys.path.append('../')
from dataset import dataloader
from utils import util, process
from dataset.dataloader import MyDataset


#%%

input_lang, output_lang, pairs = process.prepareData()


#%%
dataset = MyDataset(input_lang, output_lang, pairs, 128, True)

#%%

i=2
(in_vec, out_vec), (in_txt, out_txt) = dataset[i]
print(in_vec)
print(out_vec)
print(in_vec.shape, out_vec.shape)
print(in_txt, out_txt)