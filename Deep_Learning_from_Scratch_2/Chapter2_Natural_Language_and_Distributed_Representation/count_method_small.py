import numpy as np
from text_preprocess import preprocess
from text_create_co_matrix import create_co_matrix
from util_ppmi import ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

U, S, V = np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])
