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

2/7/2023:
* Papers:
   * Pre/Post padding: [Effects of padding on LSTMs and CNNs](https://arxiv.org/pdf/1903.07288.pdf)
   * Triangulated NMT:
      * [Ensemble Triangulation for Statistical Machine Translation](https://aclanthology.org/I13-1029.pdf) (**very similar: FR - EN**)
      * [Machine Translation by Triangulation: Making Effective Use of Multi-Parallel Corpora](https://aclanthology.org/P07-1092.pdf)
      * [Local lexical adaptation in Machine Translation through triangulation: SMT helping SMT](https://aclanthology.org/C10-1027.pdf)
      * [Revisiting Pivot Language Approach for Machine Translation](https://aclanthology.org/P09-1018.pdf)


2/6/2023:
* Save dataset, etc. w/ Pickle:
```python
with open('datafile.pkl', 'wb') as f:   # save data
  pickle.dump(dataset['train'], f)
with open('datafile.pkl', 'rb') as f:   # load data
  data = pickle.load(f)
```

* Update train_log:
```python
with open("test.txt", "a") as f: # save
   f.write("string,50,0.01,0.02")
   f.write("\n")
df = pd.read_csv('test.txt')  # read
```

* Data EDA:
   * Length: Most sentences have length of < 128 words/sent and >=5words/sent (for 3 pairs En-Fr, De-En, De-Fr w/ pkl files on Drive quan.nh) ----> only use sentences has less than 128 words (128 can be changed based on result of Tokenizer - it can be 100, then pad to 128) ----> reduce computational cost:
      * Result:

      ![Eng sent length](en-sent-len.png)
      ![Fre sent length](fr-sent-len.png)
      
      * Code:
```python
max_len_en = defaultdict(int)
max_len_fr = defaultdict(int)
for i, pair in enumerate(dataset['train']):
  pair = pair['translation']

  sent_en = pair['en']
  sent_en = sent_en.split(' ')
  max_len_en[len(sent_en)] += 1

  sent_fr = pair['fr']
  sent_fr = sent_fr.split(' ')
  max_len_fr[len(sent_fr)] += 1
  
sort_en = sorted(max_len_en.items(), key=lambda x:x[0])
sort_fr = sorted(max_len_fr.items(), key=lambda x:x[0])
sort_en_key = [key for key, val in sort_en]
sort_en_val = [val for key, val in sort_en]
sort_fr_key = [key for key, val in sort_fr]
sort_fr_val = [val for key, val in sort_fr]

plt.plot(sort_en_key, sort_en_val)
plt.plot(sort_fr_key, sort_fr_val)
```


2/5/2023:
* Tutorial:
    * [Pytorch Seq2Seq Tutorial for Machine Translation](https://www.youtube.com/watch?v=EoGUlvhRYpk) ---> [Pytorch Seq2Seq with Attention for Machine Translation](https://www.youtube.com/watch?v=sQUqQddQtB4)
* Tokenizer:
    * Spacy ([spacy.io](https://spacy.io/usage/models))
    * NLTK ([StackOverflow](https://stackoverflow.com/questions/15111183/what-languages-are-supported-for-nltk-word-tokenize-and-nltk-pos-tag))
* Add embeddings to nn.Embedding ([Medium](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76) OR [StackOverflow](https://stackoverflow.com/questions/49710537/pytorch-gensim-how-do-i-load-pre-trained-word-embeddings/49802495#49802495) OR [androidkt.com](https://androidkt.com/pre-train-word-embedding-in-pytorch/))


      

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
