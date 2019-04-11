import numpy as np
from bs4 import BeautifulSoup
from itertools import chain
from functools import partial
import tqdm
import multiprocessing

from .LegalDocument import LegalDocument as LD

def get_text_from_file(document_name):
    soup = BeautifulSoup(open(document_name), "lxml")
    temp = soup.getText()[soup.getText().find('АРБИТРАЖНЫЙ СУД МОСКОВСКОГО ОКРУГА'):]
    text = temp[:temp.find('Документ предоставлен КонсультантПлюс')]
    return text

def get_na_list_from_file(document_name):
    return LD(get_text_from_file(document_name)).na_list

def get_doc_NAs(collection, pool, vocab=False):
    doc_NAs = list(tqdm.tqdm(pool.imap(get_na_list_from_file, collection), total=len(collection)))
    return doc_NAs, np.array(list(set(chain.from_iterable(doc_NAs)))) if vocab else doc_NAs

def get_document_count_NAs(vocab, document_NAs):
    return [document_NAs.count(na) for na in vocab]

def get_docNA(collection, vocab=False):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    doc_NAs, _vocab = get_doc_NAs(collection, pool, vocab=True)
    par_get_document_count_NAs = partial(get_document_count_NAs, _vocab)
    docword = np.array(list(tqdm.tqdm(pool.imap(par_get_document_count_NAs, doc_NAs), total=len(doc_NAs))))
    return docword, _vocab if vocab else docword