"""
@Creator: Quan Nguyen
@Date: Aug 21st, 2023
@Credits: Quan Nguyen
"""

import torch
import torchtext
if torchtext.__version__ == '0.6.0':
  from .bucketdataset import get_tkzer_dict, get_field_dict
else:
  from .mydataset import get_dataset_dataloader
  
UNK_ID, PAD_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


