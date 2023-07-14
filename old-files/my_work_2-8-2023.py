# -*- coding: utf-8 -*-
"""My_work_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18f7yPhtHqXEB5KX_rKhpUiifwrRhcLQ1

## Note

Done basic steps, but how to improve performance (not good as [tutorial](https://www.youtube.com/watch?v=EoGUlvhRYpk) - non Attention):
* [ ] Apply beam search [pcyin Github](https://github.com/pcyin/pytorch_basic_nmt)
* [X] Padding base on [likarajo Github](https://github.com/likarajo/language_translation)
* [ ] Use pretrained word embedding [likarajo Github](https://github.com/likarajo/language_translation)
* [ ] Add Attention into model (next [tutorial](https://www.youtube.com/watch?v=sQUqQddQtB4))
* [ ] Use pretrained Tokenizer (Spacy)

Original paper [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf)
* [ ] Use 4 layers of LSTM
* [ ] Reversing the Source Sentences () --> how about padding?
* [X] Although LSTMs can have exploding gradients. Thus we enforced a hard constraint on the norm of the gradient [10,25] by scaling it when its norm exceeded a threshold.
* [ ]  Initialized all of the LSTM’s parameters with the uniform distribution between -0.08 and 0.08 (check [stackoverflow](https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch) OR [documen](https://pytorch.org/docs/stable/nn.init.html_))

## Setup
"""

from google.colab import drive
drive.mount('/content/gdrive')

!pip install datasets -q

# !pip install spacy -q

# !python -m spacy download fr_core_news_sm -q
# !python -m spacy download en_core_web_sm -q
# !python -m spacy download de_core_news_sm -q

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import spacy
import nltk
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""## Data

### Load data
"""

!gdown --id 19WMw9e1J7EELfTeGB0k8rIbksudEg6Kk

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.max_len = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count SOS and EOS

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
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1='eng', lang2='fra', reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    # /content/gdrive/MyDrive/Colab Notebooks/eaai24/eng-fra.txt
    lines = open('./%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 128

def filterPair(p):
  # p: a pair of lang
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
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

"""### Preprocess"""

# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]

# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.insert(0, SOS_token)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(1,-1)

# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     input_tensor = pad_sequences(input_tensor, maxlen=MAX_LENGTH, padding='pre')
#     # input_tensor = input_tensor.permute(1, 0)
#     # pad = (0, MAX_LENGTH - input_tensor.shape[1])
#     # input_tensor = F.pad(input_tensor, pad, "constant", PAD_token)

#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     target_tensor = pad_sequences(target_tensor, maxlen=MAX_LENGTH, padding='post')
#     # target_tensor = target_tensor.permute(1, 0)
#     # pad = (0, MAX_LENGTH - target_tensor.shape[1])
#     # target_tensor = F.pad(target_tensor, pad, "constant", PAD_token)
#     # output.shape = (512)
#     return (input_tensor.squeeze(), target_tensor.squeeze())

# input_lang, output_lang, pairs = prepareData('eng', 'fra', reverse=False)

# pair = random.choice(pairs)
# print(pair)
# print(len(tensorsFromPair(pair)))
# print(tensorsFromPair(pair)[0].shape, tensorsFromPair(pair)[1].shape)

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
        # padding = 'post'  # assumption from paper Seq2Seq: false

      if padding == 'pre':
        indexes = [PAD_token]*remain_len + indexes
      elif padding == 'post':
        indexes = indexes + [PAD_token]*remain_len
      else:
        indexes = indexes
      return torch.tensor(indexes, dtype=torch.long, device=device).view(-1)

  def tensorsFromPair(self, pair):
      input_tensor = self.paddingTensorFromSentence(self.input_lang, pair[0], 'pre', reverse_in=self.reverse_input)
      target_tensor = self.paddingTensorFromSentence(self.output_lang, pair[1], 'post')

      return (input_tensor, target_tensor)  # output.shape = (128)

  def __getitem__(self, index):
    pair = self.pairs[index]
    return (self.tensorsFromPair(pair), pair)

input_lang, output_lang, pairs = prepareData('eng', 'fra', False)

dataset = MyDataset(input_lang, output_lang, pairs, 10)

(en_vec, fr_vec), (en, fr) = dataset[213]
print(en, fr, en_vec.shape, fr_vec.shape)
en_vec, fr_vec

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

for (en_vec, fr_vec), (en, fr) in dataloader:
  print(en_vec.shape, fr_vec.shape)
  break

input_lang.n_words, output_lang.n_words

embedding_size = 300
hidden_size = 256
num_layers = 2
p = 0.5
embedding = nn.Embedding(input_lang.n_words, embedding_size, padding_idx=PAD_token)
dropout = nn.Dropout(p)
lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

en_vec = en_vec.squeeze().permute(1, 0)
# en_vec.shape = (seq_len, batch_size)
en_vec.shape

with torch.no_grad():
  embed = embedding(en_vec)
  outputs, (hidden, cell) = lstm(dropout(embed))
  print(embed.shape, outputs.shape, hidden.shape, cell.shape)

"""## Model

### Encoder
"""

class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    '''
    Args:
      input_size: size of Vocabulary
      embedding_size: size of vec for word2vec
      hidden_size: 1024
      num_layers: 2
      p: dropout rate = 0.5
    '''
    super(Encoder, self).__init__()
    self.dropout = nn.Dropout(p)
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.embedding = nn.Embedding(input_size, embedding_size) # output can be (batch, sent_len, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

  def forward(self, x):
    '''
    Args:
      x: has shape = (seq_len, batch_size)

    Return:
      hidden: shape = (D∗num_layers, batch_size, hidden_size if proj_size<=0 else proj_size)
      cell: shape = (D∗num_layers, bact_size, hidden_size)
    '''
    # print(f'Encoder\t x.shape = {x.shape} \t expect (512, batch_size)')
    embedding = self.dropout(self.embedding(x))
    # print(f'Encoder\t embedding.shape = {embedding.shape} \t expect (512, batch_size, 300)')

    # embedding shape = (seq_len, batch_size, embedding_size)
    # LSTM input: shape = (seq_len, batch_size, input_size)
    outputs, (hidden, cell) = self.rnn(embedding) # outputs shape: (seq_length, N, hidden_size)
    # print(f'Encoder\t hidden.shape = {hidden.shape} \t expect ({self.num_layers}, batch_size, {self.hidden_size})')
    # print(f'Encoder\t cell.shape = {cell.shape} \t expect ({self.num_layers}, batch_size, {self.hidden_size})')

    return hidden, cell # error in return shape (expect 2D)

"""### Decoder"""

class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
    '''
    input_size: size of Vocabulary
    embedding_size: size of vec for word2vec
    hidden_size: same as in Encoder
    output_size: size of Eng vocab (in case of Ger -> Eng)
    num_layers:
    p: dropout rate
    '''
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.dropout = nn.Dropout(p)
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, cell):
    '''
    Args:
      x: shape = (batch_size) because we input 1 word each time
      hidden: shape = (D * num_layers, hidden_size)
      cell: current state (for next pred)

    Return:
      pred: shape = (batch_size, target_vocab_len)
      hidden, cell: state for next pred
    '''
    # print(f'Decoder\tx.shape = {x.shape} \t expect (batch_size)')
    x = x.unsqueeze(0)  # shape = (1, batch_size) = (seq_len, batch_size) since we use a single word and not a sentence
    # print(f'Decoder\tx.shape = {x.shape} \t expect (1, batch_size)')

    embedding = self.dropout(self.embedding(x)) # embedding shape = (1, batch_size, embedding_size)
    # print(f'Decoder\t embedding.shape = {embedding.shape} \t expect (1, batch_size, 300)')
    # print(f'Decoder\t hidden.shape = {hidden.shape} \t cell.shape = {cell.shape}')
    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # outputs shape = (1, batch_size, hidden_size)
    # print(f'Decoder\t outputs.shape = {outputs.shape} \t expect (1, batch_size, {self.hidden_size})')

    predictions = self.fc(outputs)  # predictions.shape = (1, batch_size, vocab_len)
    predictions = predictions.squeeze(0)  # predictions.shape = (batch_size, target_vocab_len) to send to loss func
    # print(f'Decoder\t predictions.shape = {predictions.shape} \t expect (batch_size, target_vocab_len)')
    return predictions, hidden, cell

"""### Seq2Seq"""

class Seq2Seq(nn.Module):
  def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio=0.5):
    '''
    source: shape = (src_len, batch_size)
    target: shape = (target_len, batch_size)
    teacher_force_ratio: ratio b/w choosing predicted and ground_truth word to use as input for next word prediction
    '''
    batch_size = source.shape[1]  # need modification
    target_len = target.shape[0]  # need modification
    target_vocab_size = output_lang.n_words  # need modification (len of target vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device) # use as output prediction, init w/ zeros

    hidden, cell = self.encoder(source)

    # Grab the first input to the Decoder which will be <SOS> token
    x = target[0]
    # print(f'Seq2Seq\t start x.shape = {x.shape} \t expect (batch_size)')
    for t in range(1, target_len):
      # Use previous hidden, cell as context from encoder at start
      output, hidden, cell = self.decoder(x, hidden, cell)
      # output.shape = (batch_size, target_vocab_len)

      # print(f'Seq2Seq\t output.shape = {output.shape} \t expect (batch_size, target_vocab_len)')

      # Store next output prediction
      outputs[t] = output

      # Get the best word the Decoder predicted (index in the vocabulary)
      best_guess = output.argmax(1) # best_guess.shape = (batch_size)
      # print(f'Seq2Seq\t best_guess.shape = {best_guess.shape} \t expect (batch_size)')

      # With probability of teacher_force_ratio we take the actual next word
      # otherwise we take the word that the Decoder predicted it to be.
      # Teacher Forcing is used so that the model gets used to seeing
      # similar inputs at training and testing time, if teacher forcing is 1
      # then inputs at test time might be completely different than what the
      # network is used to. This was a long comment.
      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs

"""## Training"""

input_lang, output_lang, pairs = prepareData('eng', 'fra', False)

# Training hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = input_lang.n_words
input_size_decoder = output_lang.n_words
output_size = output_lang.n_words
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 256  # Needs to be the same for both RNN's
num_layers = 4
enc_dropout = 0.5
dec_dropout = 0.5

start_id = 500
data_len = int(6400/2)
pairs = pairs[start_id : start_id + data_len]
dataset = MyDataset(input_lang, output_lang, pairs, 128, True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

for (en_vec, fr_vec), (en, fr) in dataloader:
  print(en_vec.shape, fr_vec.shape, len(dataloader))
  print(en_vec, fr_vec)
  break

criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  # print(f"[Epoch {epoch} / {num_epochs}]")
  model.train()
  total_loss = 0.0
  for batch_idx, ((en_vec, fr_vec), (en, fr)) in tqdm(enumerate(dataloader), total=len(dataloader)):
    en_vec, fr_vec = en_vec.permute(1, 0), fr_vec.permute(1, 0) # (batch_size, seq_len) ---> (seq_len, batch_size)
    en_vec = en_vec.to(device)
    fr_vec = fr_vec.to(device)

    # Forward prop
    output = model(en_vec, fr_vec)

    # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * batch_size that we want to send in into
    # our cost function, so we need to do some reshapin. While we're at it
    # Let's also remove the start token while we're at it
    output = output[1:].reshape(-1, output.shape[2])  # shape = (trg_len * batch_size, output_dim)
    target = fr_vec[1:].reshape(-1) # shape = (trg_len * batch_size)
    # output[1:]: ignore SOS_token

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    total_loss += loss.item()
  print(f"EPOCH = {epoch} \t loss = {total_loss/len(dataloader)}")

"""DATA_SIZE = 3200, epoch = 64

---

NEW CHANGE

seq_len = 128

padding = 'pre' in (reversed) and 'post' out

encoder_embedding_size = 300

decoder_embedding_size = 300

hidden_size = 256

num_layers = 4

<details>
<summary>loss log</summary>

100%|██████████| 50/50 [01:22<00:00,  1.64s/it]
EPOCH = 0 	 loss = 6.046862239837647

100%|██████████| 50/50 [01:24<00:00,  1.68s/it]
EPOCH = 1 	 loss = 4.462330040931701

100%|██████████| 50/50 [01:22<00:00,  1.65s/it]
EPOCH = 2 	 loss = 4.330181069374085

100%|██████████| 50/50 [01:21<00:00,  1.64s/it]
EPOCH = 3 	 loss = 4.2650360584259035

100%|██████████| 50/50 [01:22<00:00,  1.64s/it]
EPOCH = 4 	 loss = 4.228246083259583

100%|██████████| 50/50 [01:22<00:00,  1.64s/it]
EPOCH = 5 	 loss = 4.1828292989730835

100%|██████████| 50/50 [01:21<00:00,  1.64s/it]
EPOCH = 6 	 loss = 4.147529201507568

100%|██████████| 50/50 [01:22<00:00,  1.64s/it]
EPOCH = 7 	 loss = 4.112067928314209

100%|██████████| 50/50 [01:23<00:00,  1.67s/it]
EPOCH = 8 	 loss = 4.0788923311233525

100%|██████████| 50/50 [01:21<00:00,  1.64s/it]
EPOCH = 9 	 loss = 4.0515174341201785

</details>

---
---

NEW CHANGE (have current the best potetial)

seq_len = 128

padding = 'pre' in (reversed) and 'post' out

encoder_embedding_size = 300

decoder_embedding_size = 300

hidden_size = 256

num_layers = 2

<details>
<summary>loss log</summary>

100%|██████████| 50/50 [01:21<00:00,  1.63s/it]
EPOCH = 0 	 loss = 6.058306541442871

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 1 	 loss = 4.408794131278992

100%|██████████| 50/50 [01:19<00:00,  1.59s/it]
EPOCH = 2 	 loss = 4.23139440536499

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 3 	 loss = 4.0753618288040165

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 4 	 loss = 3.9149249458312987

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 5 	 loss = 3.7490834188461304

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 6 	 loss = 3.5651425886154176

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 7 	 loss = 3.4091691303253175

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 8 	 loss = 3.2411407709121702

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 9 	 loss = 3.092280511856079

100%|██████████| 50/50 [01:19<00:00,  1.59s/it]
EPOCH = 10 	 loss = 2.956990485191345

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 11 	 loss = 2.8181574296951295

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 12 	 loss = 2.7334376478195193

100%|██████████| 50/50 [01:19<00:00,  1.59s/it]
EPOCH = 13 	 loss = 2.655946660041809

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 14 	 loss = 2.5171649122238158

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 15 	 loss = 2.4004191398620605

100%|██████████| 50/50 [01:20<00:00,  1.61s/it]
EPOCH = 16 	 loss = 2.27789612531662

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 17 	 loss = 2.2217716789245605

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 18 	 loss = 2.1357618045806883

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 19 	 loss = 2.0568911695480345

</details>

---
---

NEW CHANGE

seq_len = 128

padding = 'post' in (reversed) ang 'post' out

encoder_embedding_size = 300

decoder_embedding_size = 300

hidden_size = 256

num_layers = 2

<details>
<summary>loss log</summary>

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 0 	 loss = 6.034457578659057

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 1 	 loss = 4.423784399032593

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 2 	 loss = 4.2614400625228885

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 3 	 loss = 4.154506096839905

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 4 	 loss = 4.066013298034668

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 5 	 loss = 4.006021103858948

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 6 	 loss = 3.9698329877853396

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 7 	 loss = 3.929253115653992

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 8 	 loss = 3.872750663757324

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 9 	 loss = 3.8164627838134764


</details>

---
---

NEW CHANGE (have current 2nd best potetial)

seq_len = 128

padding = 'pre' and 'post' (in and out)

encoder_embedding_size = 300

decoder_embedding_size = 300

hidden_size = 256

num_layers = 2

<details>
<summary>loss log</summary>

100%|██████████| 50/50 [01:20<00:00,  1.61s/it]
EPOCH = 0 	 loss = 6.079052228927612

100%|██████████| 50/50 [01:20<00:00,  1.61s/it]
EPOCH = 1 	 loss = 4.4093651819229125

100%|██████████| 50/50 [01:21<00:00,  1.62s/it]
EPOCH = 2 	 loss = 4.234301209449768

100%|██████████| 50/50 [01:21<00:00,  1.64s/it]
EPOCH = 3 	 loss = 4.087865462303162

100%|██████████| 50/50 [01:20<00:00,  1.62s/it]
EPOCH = 4 	 loss = 3.977463812828064

100%|██████████| 50/50 [01:21<00:00,  1.63s/it]
EPOCH = 5 	 loss = 3.8644219923019407

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 6 	 loss = 3.6924276685714723

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 7 	 loss = 3.5700292873382566

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 8 	 loss = 3.3947692918777466

100%|██████████| 50/50 [01:18<00:00,  1.57s/it]
EPOCH = 9 	 loss = 3.2692558479309084

100%|██████████| 50/50 [01:21<00:00,  1.63s/it]
EPOCH = 10 	 loss = 3.1830705881118773

100%|██████████| 50/50 [01:20<00:00,  1.61s/it]
EPOCH = 11 	 loss = 3.098177933692932

100%|██████████| 50/50 [01:20<00:00,  1.62s/it]
EPOCH = 12 	 loss = 3.017664046287537

100%|██████████| 50/50 [01:21<00:00,  1.63s/it]
EPOCH = 13 	 loss = 2.8849183225631716

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 14 	 loss = 2.768659610748291

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 15 	 loss = 2.6834586668014526

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 16 	 loss = 2.6142616200447084

100%|██████████| 50/50 [01:18<00:00,  1.58s/it]
EPOCH = 17 	 loss = 2.5200012016296385

100%|██████████| 50/50 [01:19<00:00,  1.58s/it]
EPOCH = 18 	 loss = 2.476028332710266

100%|██████████| 50/50 [01:19<00:00,  1.60s/it]
EPOCH = 19 	 loss = 2.35499146938324

</details>

---
---

NEW CHANGE

seq_len = 128

padding = 'pre' and 'post' (in and out)

encoder_embedding_size = 100

decoder_embedding_size = 100

hidden_size = 128

num_layers = 1


<details>
<summary>loss log</summary>

100%|██████████| 50/50 [01:14<00:00,  1.48s/it]
EPOCH = 0 	 loss = 7.378298559188843

100%|██████████| 50/50 [01:12<00:00,  1.46s/it]
EPOCH = 1 	 loss = 4.4149853801727295

100%|██████████| 50/50 [01:12<00:00,  1.45s/it]
EPOCH = 2 	 loss = 4.219454731941223

100%|██████████| 50/50 [01:12<00:00,  1.45s/it]
EPOCH = 3 	 loss = 4.088070254325867

100%|██████████| 50/50 [01:16<00:00,  1.53s/it]
EPOCH = 4 	 loss = 4.004882974624634

100%|██████████| 50/50 [01:15<00:00,  1.50s/it]
EPOCH = 5 	 loss = 3.9452049922943115

100%|██████████| 50/50 [01:13<00:00,  1.47s/it]
EPOCH = 6 	 loss = 3.8773948192596435

100%|██████████| 50/50 [01:14<00:00,  1.48s/it]
EPOCH = 7 	 loss = 3.804203724861145

100%|██████████| 50/50 [01:14<00:00,  1.50s/it]
EPOCH = 8 	 loss = 3.709712514877319

100%|██████████| 50/50 [01:13<00:00,  1.48s/it]
EPOCH = 9 	 loss = 3.6215027523040773

</details>

---
---

seq_len = 512

padding = 'post'

encoder_embedding_size = 300

decoder_embedding_size = 300

hidden_size = 256

num_layers = 2


<details>
<summary>loss log</summary>

100%|██████████| 50/50 [07:26<00:00,  8.94s/it]
EPOCH = 0 	 loss = 5.383958940505981

100%|██████████| 50/50 [07:25<00:00,  8.90s/it]
EPOCH = 1 	 loss = 3.7594539594650267

100%|██████████| 50/50 [07:25<00:00,  8.90s/it]
EPOCH = 2 	 loss = 3.5973013925552366

100%|██████████| 50/50 [07:24<00:00,  8.90s/it]
EPOCH = 3 	 loss = 3.478486728668213

100%|██████████| 50/50 [07:25<00:00,  8.90s/it]
EPOCH = 4 	 loss = 3.4210835123062133

100%|██████████| 50/50 [07:24<00:00,  8.88s/it]
EPOCH = 5 	 loss = 3.356914539337158

100%|██████████| 50/50 [07:24<00:00,  8.90s/it]
EPOCH = 6 	 loss = 3.2783570289611816

100%|██████████| 50/50 [07:23<00:00,  8.87s/it]
EPOCH = 7 	 loss = 3.2372921180725096

100%|██████████| 50/50 [07:24<00:00,  8.89s/it]
EPOCH = 8 	 loss = 3.199459252357483

100%|██████████| 50/50 [07:24<00:00,  8.89s/it]
EPOCH = 9 	 loss = 3.1788624906539917

100%|██████████| 50/50 [07:23<00:00,  8.87s/it]
EPOCH = 10 	 loss = 3.1831029272079467

100%|██████████| 50/50 [07:24<00:00,  8.88s/it]
EPOCH = 11 	 loss = 3.1025032949447633

100%|██████████| 50/50 [07:24<00:00,  8.88s/it]
EPOCH = 12 	 loss = 3.1095044803619385

100%|██████████| 50/50 [07:23<00:00,  8.86s/it]
EPOCH = 13 	 loss = 3.0869728183746337

100%|██████████| 50/50 [07:22<00:00,  8.86s/it]
EPOCH = 14 	 loss = 3.08543803691864

100%|██████████| 50/50 [07:22<00:00,  8.84s/it]
EPOCH = 15 	 loss = 3.0724154567718505

100%|██████████| 50/50 [07:21<00:00,  8.83s/it]
EPOCH = 16 	 loss = 3.0756219959259035

100%|██████████| 50/50 [07:22<00:00,  8.85s/it]
EPOCH = 17 	 loss = 3.017907304763794

100%|██████████| 50/50 [07:21<00:00,  8.84s/it]
EPOCH = 18 	 loss = 3.0001839065551756

100%|██████████| 50/50 [07:22<00:00,  8.85s/it]
EPOCH = 19 	 loss = 3.021025981903076

100%|██████████| 50/50 [07:22<00:00,  8.84s/it]
EPOCH = 20 	 loss = 2.9646836137771606

100%|██████████| 50/50 [07:22<00:00,  8.84s/it]
EPOCH = 21 	 loss = 2.98474328994751

 28%|██▊       | 14/50 [02:12<05:40,  9.47s/it]

 </details>

## Eval
"""

def translate_sentence(model, en_vec, output_lang, device, max_length=50):
  model.eval()
  vec = en_vec[0]
  vec = vec.unsqueeze(0)

  # Build encoder hidden, cell state
  with torch.no_grad():
      hidden, cell = model.encoder(vec)

  outputs = [SOS_token]

  for _ in range(max_length):
      previous_word = torch.LongTensor([outputs[-1]]).to(device)

      with torch.no_grad():
          output, hidden, cell = model.decoder(previous_word, hidden, cell)
          best_guess = output.argmax(1).item()

      outputs.append(best_guess)

      # Model predicts it's the end of the sentence
      if output.argmax(1).item() == EOS_token:
          break

  print(outputs)
  translated_sentence = [output_lang.index2word[idx] for idx in outputs]

  # remove start token
  return translated_sentence

testset = MyDataset(input_lang, output_lang, pairs[-651:], 128, True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1)

def translate_sentence(model, en_vec, output_lang, device, max_length=50):
  model.eval()
  vec = en_vec[0]
  vec = vec.unsqueeze(0)
  vec = vec.permute(1, 0)
  # print(vec.shape)

  # Build encoder hidden, cell state
  with torch.no_grad():
      hidden, cell = model.encoder(vec)
      # print(hidden.shape, cell.shape)

  outputs = [SOS_token]

  for _ in range(max_length):
      previous_word = torch.LongTensor([outputs[-1]]).to(device)

      with torch.no_grad():
          output, hidden, cell = model.decoder(previous_word, hidden, cell)
          best_guess = output.argmax(1).item()

      outputs.append(best_guess)

      # Model predicts it's the end of the sentence
      if output.argmax(1).item() == EOS_token:
          break

  translated_sentence = [output_lang.index2word[idx] for idx in outputs]
  print(translated_sentence)
  return translated_sentence

for idx, ((en_vec, fr_vec), (en, fr)) in enumerate(testloader):
  # print(en_vec.shape, fr_vec.shape)
  # print(en[0], en_vec[0][:10])
  print(fr[0])
  translate_sentence(model, en_vec, output_lang, device, max_length=50)
  if idx==50: break