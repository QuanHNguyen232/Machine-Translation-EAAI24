#%%
from torch.nn.utils.rnn import pad_sequence
import torch
from models import Seq2SeqTransformer
from utils import util

#%%
S = 5
T = 3
N = 4
E = 16

x1 = torch.ones(S, N)
x2 = torch.ones(T, N) * 2

print(x1)
print(x2)

z = pad_sequence([x1, x2]) # (long_seq, stack, N)
print(z.shape)
print((x1 == z[:, 0, :]).all())
print(z[:, 1, :])
print(x2)

#%%
cfg = util.load_cfg()
device = cfg['device']

cfg['seq2seq']['en_DIM'] = 100
cfg['seq2seq']['fr_DIM'] = 200
model = Seq2SeqTransformer(cfg, 'en', 'fr', 1, device)

#%%
UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


src1 = torch.tensor([[SOS_ID, 8, EOS_ID, PAD_ID, PAD_ID], [SOS_ID, 10, 20, 30, EOS_ID]]).permute(1, 0)
trg1 = torch.tensor([[SOS_ID, 8, EOS_ID, PAD_ID, PAD_ID], [SOS_ID, 10, 20, 30, EOS_ID]]).permute(1, 0)

src2 = torch.tensor([[SOS_ID, EOS_ID, PAD_ID, PAD_ID, PAD_ID], [SOS_ID, 10, 20, EOS_ID, PAD_ID]]).permute(1, 0)
trg2 = torch.tensor([[SOS_ID, 8, 10, 12, EOS_ID], [SOS_ID, 10, EOS_ID, PAD_ID, PAD_ID]]).permute(1, 0)



#%%
print(src1.shape)
print(src1)

src_mask1, tgt_mask1, src_padding_mask1, tgt_padding_mask1 = model.create_mask(src1, trg1[:-1, :])
print(src_mask1)
print(tgt_mask1)
print(src_padding_mask1)
print(tgt_padding_mask1)

#%%
print(src2.shape)
print(src2)

src_mask2, tgt_mask2, src_padding_mask2, tgt_padding_mask2 = model.create_mask(src2, trg2[:-1, :])
print(src_mask2)
print(tgt_mask2)
print(src_padding_mask2)
print(tgt_padding_mask2)


#%%

print(src_padding_mask1)
print(src_padding_mask2)

combine = ~torch.logical_or(~src_padding_mask1, ~src_padding_mask2) # ~: invert
