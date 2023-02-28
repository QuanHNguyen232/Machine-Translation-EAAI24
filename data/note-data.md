# Notes for using data

* USING datasets (generated from `Filtering dataset.ipynb`):
  * en-fr: `enfr_160kpairs_2k5-freq-words.pkl`
  * en-de-fr: `endefr_75kpairs_2k5-freq-words.pkl`
* USING vocab index - must match to have models perform well for torchtext.Field ([instruction](https://discuss.pytorch.org/t/how-to-save-and-load-torchtext-data-field-build-vocab-result/50407/3)):
  * Obtain from first 64k pairs from `endefr_75kpairs_2k5-freq-words.pkl`:
    1. `piv-endefr-src_vocab.txt`
    1. `piv-endefr-piv_vocab.txt`
    1. `piv-endefr-trg_vocab.txt`
  * Obtain from first 64k pairs from `enfr_160kpairs_2k5-freq-words.pkl`:
    1. `seq2seq-enfr-src_vocab.txt`
    1. `seq2seq-enfr-trg_vocab.txt`

<details>
<summary>How to get data from HuggingFace</summary>

* Install package
  ```shell
  pip install datasets -q
  ```

* Download data from hugging face ([europarl_bilingual](https://huggingface.co/datasets/europarl_bilingual))
  ```python
  from datasets import list_datasets, load_dataset
  print('europarl_bilingual' in list_datasets())

  lang_1, lang_2 = 'en', 'fr'
  dataset = load_dataset("europarl_bilingual", lang1=lang_1, lang2=lang_2)
  ```

* Convert data type dataset (huggingFace) to Python list:
  ```python
  lang_1, lang_2 = 'en', 'fr'
  mydata = []
  for pair in dataset['train']:
    pair = pair['translation']
    curr_pair = {
        lang_1: pair[lang_1],
        lang_2: pair[lang_2]
    }
    mydata.append(curr_pair)
  ```

* Save data to pickle as `.pkl` file:
  ```python
  import pickle

  lang_1, lang_2 = 'en', 'fr'
  with open(f'{lang_1}-{lang_2}.pkl', 'wb') as f:
    pickle.dump(mydata, f)
  ```

* Load data from pickle (`.pkl` file):
  ```python
  import pickle

  lang_1, lang_2 = 'en', 'fr'
  with open(f'{lang_1}-{lang_2}.pkl', 'rb') as f:
    data = pickle.load(f)
  ```
</details>