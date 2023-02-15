"""
@Creator: Quan Nguyen
@Date: Feb 8, 2023
@Credits: Quan Nguyen

dataloader file in dataset
"""

import torch
from typing import List, Tuple

from utils.process import Lang


PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 128


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, input_lang: Lang, output_lang: Lang, pairs: Tuple, max_len: int=128, reverse_input: bool=False):
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.pairs = pairs
    self.MAX_LENGTH = max_len
    self.reverse_input = reverse_input
  
  def __len__(self):
    return len(self.pairs)
  
  def indexesFromSentence(self, lang: Lang, sentence: str):
    ''' convert sentence to vector by word index (add <sos> and <eos> tags)
    Args:
      lang: language (Lang obj) has word2id, id2word dicts
      sentence: sentence wanted to be converted to index
    Return:
      List: indices of words
    '''
    return [SOS_token] + [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

  def paddingTensorFromSentence(self, lang: Lang, sentence: str, padding: str='pre', reverse_in: bool=False):
    ''' Add padding to each sentence
    Args:
      lang: Lang object
      sentence:
      padding: choosing b/w pre/post padding
      reverse_in: whether to reverse input
    Return:
      Tensor: torch.tensor of indices
    '''
    indexes = self.indexesFromSentence(lang, sentence)
    remain_len = self.MAX_LENGTH - len(indexes)
    
    if reverse_in:
      indexes = list(reversed(indexes))

    if padding == 'pre':
      indexes = [PAD_token]*remain_len + indexes
    else:
      indexes = indexes + [PAD_token]*remain_len

    return torch.tensor(indexes, dtype=torch.long).view(-1)

  def tensorsFromPair(self, pair: Tuple):
    '''
    Args:
      pair: each pair of language
    Return:
      input_tensor: tensor of input language
      target_tensor: tensor of output language
    '''
    input_tensor = self.paddingTensorFromSentence(self.input_lang, pair[0], 'pre', reverse_in=self.reverse_input)
    target_tensor = self.paddingTensorFromSentence(self.output_lang, pair[1], 'post')
    
    return (input_tensor, target_tensor)  # output.shape = (128)

  def __getitem__(self, index: int):
    pair = self.pairs[index]
    return (self.tensorsFromPair(pair), pair)
