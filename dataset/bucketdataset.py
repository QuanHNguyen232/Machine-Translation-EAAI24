
import spacy

import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data import Dataset, Example


def get_tkzer_dict(langs):
    tkz_dict = {}
    if 'en' in langs:
        spacy_en = spacy.load('en_core_web_sm')
        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]
        tkz_dict['en'] = tokenize_en
    if 'de' in langs:
        spacy_de = spacy.load('de_core_news_sm')
        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]
        tkz_dict['de'] = tokenize_de
    if 'fr' in langs:
        spacy_fr = spacy.load('fr_core_news_sm')
        def tokenize_fr(text):
            return [tok.text for tok in spacy_fr.tokenizer(text)]
        tkz_dict['fr'] = tokenize_fr
    if 'it' in langs:
        spacy_it = spacy.load('it_core_news_sm')
        def tokenize_it(text):
            return [tok.text for tok in spacy_it.tokenizer(text)]
        tkz_dict['it'] = tokenize_it
    if 'es' in langs:
        spacy_es = spacy.load('es_core_news_sm')
        def tokenize_es(text):
            return [tok.text for tok in spacy_es.tokenizer(text)]
        tkz_dict['es'] = tokenize_es
    if 'pt' in langs:
        spacy_pt = spacy.load('pt_core_news_sm')
        def tokenize_pt(text):
            return [tok.text for tok in spacy_pt.tokenizer(text)]
        tkz_dict['pt'] = tokenize_pt
    if 'ro' in langs:
        spacy_ro = spacy.load('ro_core_news_sm')
        def tokenize_ro(text):
            return [tok.text for tok in spacy_ro.tokenizer(text)]
        tkz_dict['ro'] = tokenize_ro
    return tkz_dict

def get_field_dict(tkzer_dict):
    field_dict = {}
    for lang in tkzer_dict.keys():
        field_dict[lang] = Field(tokenize = tkzer_dict[lang], init_token = '<sos>', eos_token = '<eos>', lower = True, include_lengths = True)
    return field_dict