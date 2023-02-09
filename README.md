# Machine-Translation-EAAI24

TO DO:
* [ ] ERROR in Seq2Seq model, but why it stil can run??? ([`models/base.py`](models/base.py))
* [ ] Apply beam search [pcyin Github](https://github.com/pcyin/pytorch_basic_nmt)
* [X] Padding base on [likarajo Github](https://github.com/likarajo/language_translation)
* [ ] Use pretrained word embedding [likarajo Github](https://github.com/likarajo/language_translation)
* [ ] Add Attention into model (next [tutorial](https://www.youtube.com/watch?v=sQUqQddQtB4))
* [ ] Use pretrained Tokenizer (Spacy)

* Original paper [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf):
    * [ ] Use 4 layers of LSTM
    * [X] Reversing the Source Sentences () --> output should be like this:
        ```python
        PAD_token = 0
        SOS_token = 1
        EOS_token = 2
        in_vec  = [0, 0, 0, 0, 0, 0, 2, 10, 12, 23, 35, 1]
        out_vec = [1, 32, 14, 54, 31, 2, 0, 0, 0, 0, 0, 0]
        ```

    * [X] Although LSTMs can have exploding gradients. Thus we enforced a hard constraint on the norm of the gradient [10,25] by scaling it when its norm exceeded a threshold.
    * [ ] Initialized all of the LSTM’s parameters with the uniform distribution between -0.08 and 0.08 (check [stackoverflow](https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch) OR [documen](https://pytorch.org/docs/stable/nn.init.html_))

Best performance till now:
```python
# padding = 'pre' in (reversed) and 'post' out
seq_len = 128
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 256
num_layers = 2
```