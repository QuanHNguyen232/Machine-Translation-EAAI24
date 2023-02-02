Most prefer:
* Pretrained Model: mbart - hugging face ([link]())
* Data: EuroParl - hugging face ([link](https://huggingface.co/datasets/europarl_bilingual))
* Metrics: BLEU ([huggingFace](https://huggingface.co/spaces/evaluate-metric/bleu) or [personal code](https://github.com/ymoslem/MT-Evaluation/blob/main/BLEU/compute-bleu.py))

---
Next week:
* Try merging systems + continuous system
    * first try direct translate + metric (based level) (focus more on basic **translating machine**)
    * how to make merging system feasible (critical problem that must be solved)
* Try metrics systems
---

2/2/2023:
* Word embeddings:
    * [fasttext: multi-lingual word vectors](https://fasttext.cc/docs/en/crawl-vectors.html) or [Github](https://github.com/facebookresearch/fastText/tree/master)
    * [Fb MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE#multilingual-word-embeddings)
* Model:
    * [OpenNMT-py (Github)](https://github.com/OpenNMT/OpenNMT-py) (designed to be research friendly to try out new ideas in translation)
        * -> [Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial)
* Metric: BLEU: [huggingFace](https://huggingface.co/spaces/evaluate-metric/bleu) OR [ymoslem/MT-Evaluation (Github)](https://github.com/ymoslem/MT-Evaluation/blob/main/BLEU/compute-bleu.py)

2/1/2023:
* [simple Seq2Seq w/ Attention (Pytorch)](https://github.com/graykode/nlp-tutorial)
* Choosing languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish) ([europarl group](https://www.statmt.org/europarl/))


1/30/2023
* Dataset: https://huggingface.co/datasets/europarl_bilingual (21 languages) - only has train set, download directly from [Europarl](https://www.statmt.org/europarl/) otherwise. Command:
```python
!pip install datasets
from datasets import list_datasets, load_dataset
print('europarl_bilingual' in list_datasets())
dataset = load_dataset("europarl_bilingual", lang1="en", lang2="fr")  # https://huggingface.co/datasets/europarl_bilingual
```
* Models:
    * [mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) -> for low-resource languages (e.g. a few thousands to a few millions, up to 15m), using directly or fine-tuning mBART can give better results ([link](https://blog.machinetranslation.io/multilingual-nmt/))
    * [Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
    * List of models by [OpusMT](https://opus.nlpl.eu/Opus-MT/)
    * Most are Transformers, I want to find basic models with RNNs, otherwise, I have to build myself:
        * [Pytorch tutorial Seq2Seq](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model)
        * [Github: pcyin/pytorch_basic_nmt](https://github.com/pcyin/pytorch_basic_nmt)
        * [Github: marumalo/pytorch-seq2seq](https://github.com/marumalo/pytorch-seq2seq)
        * [blog.paperspace.com seq2seq pytorch](https://blog.paperspace.com/seq2seq-translator-pytorch/)
        * [Medium saikrishna4820/lstm-language-translation](https://medium.com/@saikrishna4820/lstm-language-translation-18c076860b23)
        * [TowardsDatScience: how to build an encoder decoder translation model using lstm with python and keras](https://towardsdatascience.com/how-to-build-an-encoder-decoder-translation-model-using-lstm-with-python-and-keras-a31e9d864b9b)
        * Base on this: [Github likarajo/language_translation](https://github.com/likarajo/language_translation)
        * Base on this: [Language Translator (RNN BiDirectional LSTMs and Attention) in Python](https://www.codespeedy.com/language-translator-rnn-bidirectional-lstms-and-attention-in-python/)