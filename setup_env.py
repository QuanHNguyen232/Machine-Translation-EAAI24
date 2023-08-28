import subprocess

# install spacy tokenizers
subprocess.run('python -m spacy download en_core_web_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download de_core_news_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download fr_core_news_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download it_core_news_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download es_core_news_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download pt_core_news_sm -q', shell=True, check=True)
subprocess.run('python -m spacy download ro_core_news_sm -q', shell=True, check=True)
subprocess.run('!python -m spacy validate', shell=True, check=True)