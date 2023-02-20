
import torch
import torch.nn as nn
from torch.utils import data

from tqdm import tqdm



def train_fn(model: nn.Module, dataloader: data.DataLoader, optimizer: torch.optim, criterion, cfg: dict):
  model.train()
  total_loss = 0.0
  for (en_vec, fr_vec), (en, fr) in tqdm(dataloader):
    en_vec, fr_vec = en_vec.permute(1, 0), fr_vec.permute(1, 0) # (N, seq_len) ---> (seq_len, N)
    en_vec = en_vec.to(cfg.get('device', 'cpu'))
    fr_vec = fr_vec.to(cfg.get('device', 'cpu'))

    # Forward prop
    output = model(en_vec, fr_vec)  # (tgt_len, N, output_dim)

    # Output is of shape (trg_len, N, output_dim) but Cross Entropy Loss
    # doesn't take input in that form. For example if we have MNIST we want to have
    # output to be: (N, 10) and targets just (N). Here we can view it in a similar
    # way that we have output_words * N that we want to send in into
    # our cost function, so we need to do some reshapin. While we're at it
    # Let's also remove the start token while we're at it
    output = output[1:].reshape(-1, output.shape[2])  # shape = (trg_len * N, output_dim)
    target = fr_vec[1:].reshape(-1) # shape = (trg_len * N)
    # output[1:]: ignore SOS_token

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    
    total_loss += loss.item()

  return total_loss/len(dataloader)


def eval_fn(model: nn.Module, dataloader: data.DataLoader, criterion, cfg: dict):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for (en_vec, fr_vec), (en, fr) in tqdm(dataloader):
        en_vec, fr_vec = en_vec.permute(1, 0), fr_vec.permute(1, 0) # (N, seq_len) ---> (seq_len, N)
        en_vec = en_vec.to(cfg.get('device', 'cpu'))
        fr_vec = fr_vec.to(cfg.get('device', 'cpu'))

        # Forward prop
        output = model(en_vec, fr_vec)  # (tgt_len, N, output_dim)

        # Output is of shape (trg_len, N, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * N that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])  # shape = (trg_len * N, output_dim)
        target = fr_vec[1:].reshape(-1) # shape = (trg_len * N)
        # output[1:]: ignore SOS_token

        loss = criterion(output, target)
        
        total_loss += loss.item()

    return total_loss/len(dataloader)