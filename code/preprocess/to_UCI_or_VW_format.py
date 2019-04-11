from .get_docword import get_docword
from .get_docNA import get_docNA
import numpy as np

def cut_most_fr(docword, vocab, cut_most_fr_procent):
    most_fr_words = docword.sum(axis=0).argsort()[::-1][:int(len(vocab)*0.01*cut_most_fr_procent)]
    return np.delete(docword, most_fr_words, axis=1), np.delete(vocab, most_fr_words)

def to_UCI(collection, collection_name, cut_most_fr_procent=0):
    docword, vocab = get_docword(collection, vocab=True)
    if cut_most_fr_procent:
        docword, vocab = cut_most_fr(docword, vocab, cut_most_fr_procent)
    NNZ = []
    for i in range(len(docword)):
        for j in range(len(docword[i])):
            if docword[i][j]:
                NNZ.append('{} {} {}'.format(i+1, j+1, docword[i][j]))
    with open('./docword.' + collection_name + '.txt', 'w') as docword_file:
        docword_file.write('{}\n{}\n{}\n'.format(len(docword), len(vocab), len(NNZ)) + '\n'.join(NNZ))
    with open('./vocab.' + collection_name + '.txt', 'w') as vocab_file:
        vocab_file.write('\n'.join(vocab))
        
def cut_most_fr_NA(docNA, vocab, cut_most_fr_procent):
    most_fr_words = docNA.sum(axis=0).argsort()[::-1][:int(len(vocab)*0.01*cut_most_fr_procent)]
    return np.delete(docNA, most_fr_words, axis=1), np.delete(vocab, most_fr_words)
        
def to_VW(collection, collection_name, cut_most_fr_procent=0, cut_most_fr_procent_NA=0):
    docword, vocab = get_docword(collection, vocab=True)
    docNA, vocab_NA = get_docNA(collection, vocab=True)
    if cut_most_fr_procent:
        docword, vocab = cut_most_fr(docword, vocab, cut_most_fr_procent)
    if cut_most_fr_procent:
        docNA, vocab_NA = cut_most_fr_NA(docNA, vocab_NA, cut_most_fr_procent_NA)
    dict_vocab = {i : vocab[i] for i in range(len(vocab))}
    default_class_part = ['doc_{} |@default_class '.format(j) + ' '.join(['{}:{}'.format(dict_vocab[i], docword[j,i]) for i in range(len(docword[j])) if docword[j,i]]) for j in range(len(docword))]
    na_class_part = [' |@na_class '.format(j) + ' '.join(['NA_{}:{}'.format(i, docNA[j,i]) for i in range(len(docNA[j])) if docNA[j,i]]) for j in range(len(docNA))]

    with open('./vw.' + collection_name + '.txt', 'w') as vw_file:
        vw_file.write('\n'.join([default_class_part[i]+na_class_part[i] for i in range(len(collection))]))

