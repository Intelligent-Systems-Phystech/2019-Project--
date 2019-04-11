from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
from bs4 import BeautifulSoup
import numpy as np
import multiprocessing
import tqdm
from functools import partial
from itertools import chain


def get_preprocessed_BoW_from_text(text):
    morph = pymorphy2.MorphAnalyzer()
    tokenized_text = RegexpTokenizer(r'[а-яА-Я]+\-?[а-яА-Я]+').tokenize(text.replace('ё','е'))
    return [morph.parse(word.lower())[0].normal_form for word in tokenized_text if morph.parse(word.lower())[0].normal_form not in stopwords.words('russian') and len(word)>2]

def get_preprocessed_BoW_from_file(document_name):
    soup = BeautifulSoup(open(document_name), "lxml")
    temp = soup.getText()[soup.getText().find('АРБИТРАЖНЫЙ СУД МОСКОВСКОГО ОКРУГА'):]
    text = temp[:temp.find('Документ предоставлен КонсультантПлюс')]
    return get_preprocessed_BoW_from_text(text)

def get_tokenized_texts(collection, pool, vocab=False):
    tokenized_texts = np.array(list(tqdm.tqdm(pool.imap(get_preprocessed_BoW_from_file, collection), total=len(collection))))
    return tokenized_texts, np.array(sorted(set(chain.from_iterable(tokenized_texts)))) if vocab else tokenized_texts

def get_document_word(vocab, tokenized_text):
    return [tokenized_text.count(word) for word in vocab]

def get_docword(collection, vocab=False):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    tokenized_texts, _vocab = get_tokenized_texts(collection, pool, vocab=True)
    par_get_document_word = partial(get_document_word, _vocab)
    docword = np.array(list(tqdm.tqdm(pool.imap(par_get_document_word, tokenized_texts), total=len(tokenized_texts))))
    return docword, _vocab if vocab else docword