"""
@Creator: Quan Nguyen
@Date: Feb 8, 2023
@Credits: Quan Nguyen

dataloader file in dataset
"""

import torch


PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 128


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, input_lang, output_lang, pairs, max_len=128, reverse_input=False):
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.pairs = pairs
    self.MAX_LENGTH = max_len
    self.reverse_input = reverse_input
  
  def __len__(self):
    return len(self.pairs)
  
  def indexesFromSentence(self, lang, sentence):
    return [SOS_token] + [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]

  def paddingTensorFromSentence(self, lang, sentence, padding='pre', reverse_in=False):
      indexes = self.indexesFromSentence(lang, sentence)
      remain_len = self.MAX_LENGTH - len(indexes)
      
      if reverse_in:
        indexes = list(reversed(indexes))

      if padding == 'pre':
        indexes = [PAD_token]*remain_len + indexes
      else:
        indexes = indexes + [PAD_token]*remain_len

      return torch.tensor(indexes, dtype=torch.long).view(-1)

  def tensorsFromPair(self, pair):
      input_tensor = self.paddingTensorFromSentence(self.input_lang, pair[0], 'pre', reverse_in=self.reverse_input)
      target_tensor = self.paddingTensorFromSentence(self.output_lang, pair[1], 'post')
      
      return (input_tensor, target_tensor)  # output.shape = (128)

  def __getitem__(self, index):
    pair = self.pairs[index]
    return (self.tensorsFromPair(pair), pair)
