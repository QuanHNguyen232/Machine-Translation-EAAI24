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
  def __init__(self, input_lang, output_lang, pairs, cfg):
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.pairs = pairs
    self.cfg = cfg
  
  def __len__(self):
    return len(self.pairs)
  
  def tokenizeTxt(self, lang, text):  # Tokenizes text from a string into a list of strings (tokens)
    if lang.name == self.input_lang.name:
      return [tok.text for tok in self.input_lang.tkz.tokenizer(text)]  # self.in_tkz.tokenizer(text)
    return [tok.text for tok in self.output_lang.tkz.tokenizer(text)]  # self.out_tkz.tokenizer(text)

  def indexesFromSentence(self, lang, sentence):
    ''' convert sentence to vector by word index (add <sos> and <eos> tags)
    Args:
      lang: language (Lang obj) has word2id, id2word dicts
      sentence: sentence wanted to be converted to index
    Return:
      List: indices of words
    '''
    words = self.tokenizeTxt(lang, sentence)
    sent2id = [lang.word2index[word] for word in words]
    return [self.cfg['SOS_token']] + sent2id[:self.cfg['max_seq_len']-2] + [self.cfg['EOS_token']]  # 2 for <sos> and <eos>

  def paddingTensorFromSentence(self, lang, sentence, padding, reverse_in):
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
    remain_len = self.cfg['max_seq_len'] - len(indexes)
    if reverse_in:
      indexes = list(reversed(indexes))
    if padding == 'pre':
      indexes = [self.cfg['PAD_token']]*remain_len + indexes
    elif padding == 'post':
      indexes = indexes + [self.cfg['PAD_token']]*remain_len
    
    return torch.tensor(indexes, dtype=torch.long).view(-1) # output.shape = (cfg['max_seq_len']) = [64]

  def tensorsFromPair(self, pair):
    '''
    Args:
      pair: each pair of language
    Return:
      input_tensor: tensor of input language
      target_tensor: tensor of output language
    '''
    input_tensor = self.paddingTensorFromSentence(self.input_lang, pair[0], self.cfg['input_pad'], reverse_in=self.cfg['input_reverse'])
    target_tensor = self.paddingTensorFromSentence(self.output_lang, pair[1], 'post', reverse_in=False)
    
    return (input_tensor, target_tensor)  # output.shape = (cfg['max_seq_len']) = [64]

  def __getitem__(self, index):
    pair = self.pairs[index]
    return self.tensorsFromPair(pair), pair