# Machine-Translation-EAAI24

**NOTE**:
* Main pipeline: `bentrevett_pytorch_seq2seq.ipynb`
* Pivot model EN-DE-FR: `piv_endefr_74kset.pt`
* `attn_en-fr_32k_160kset_inverse.pt` achieves BLEU=32.15
    * Dataset: 160kpairs-2k5-freq-words
    * Inverse input (inverse then add `<sos>`, `<eos>`, and `<pad>` (post padding))
    * Batch=32
    * Epochs = 10
    * train_len = 64000, test_len = 3200
    * ENC_EMB_DIM = 256, DEC_EMB_DIM = 256, ENC_HID_DIM = 512, DEC_HID_DIM = 512, ENC_DROPOUT = 0.5, DEC_DROPOUT = 0.5, LR = 0.001
    * init_weights: nn.init.normal_(param.data, mean=0, std=0.01)


* Original paper [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf):
    <!-- * [ ] Use 4 layers of LSTM -->
    * [X] Reversing the Source Sentences () --> output should be like this:
        ```python
        PAD_token = 0
        SOS_token = 1
        EOS_token = 2
        in_vec  = [0, 0, 0, 0, 0, 0, 2, 10, 12, 23, 35, 1]
        out_vec = [1, 32, 14, 54, 31, 2, 0, 0, 0, 0, 0, 0]
        ```

    * [X] Although LSTMs can have exploding gradients. Thus we enforced a hard constraint on the norm of the gradient [10,25] by scaling it when its norm exceeded a threshold.
    * [X] Initialized all of the LSTM’s parameters with the uniform distribution between -0.08 and 0.08 (check [stackoverflow](https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch) OR [documen](https://pytorch.org/docs/stable/nn.init.html_))
    * [ ] Use pad_pack to reduce computation:
        * https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        * https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/5
        * https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a
Best performance till now:
```python
# padding = 'pre' in (reversed) and 'post' out
seq_len = 128
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 256
num_layers = 2
```