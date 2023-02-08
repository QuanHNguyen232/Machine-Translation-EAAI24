How to get data from HuggingFace:

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

Save data to pickle as `.pkl` file:
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