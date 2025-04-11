import argparse
import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn

# --------- Argument Parsing ---------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
args = parser.parse_args()
dataset = args.dataset

# --------- Utility Functions ---------
def read_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def write_lines(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

# --------- Read Document Names and Contents ---------
doc_names = read_lines(f'data/{dataset}.txt')
doc_contents = read_lines(f'data/corpus/{dataset}.clean.txt')

doc_train, doc_test = [], []
for name in doc_names:
    temp = name.split("\t")
    if 'train' in temp[1]:
        doc_train.append(name)
    elif 'test' in temp[1]:
        doc_test.append(name)

train_ids = [doc_names.index(name) for name in doc_train]
test_ids = [doc_names.index(name) for name in doc_test]

random.shuffle(train_ids)
random.shuffle(test_ids)

write_lines(f'data/{dataset}.train.index', map(str, train_ids))
write_lines(f'data/{dataset}.test.index', map(str, test_ids))

ids = train_ids + test_ids
shuffle_doc_names = [doc_names[i] for i in ids]
shuffle_doc_words = [doc_contents[i] for i in ids]

write_lines(f'data/{dataset}_shuffle.txt', shuffle_doc_names)
write_lines(f'data/corpus/{dataset}_shuffle.txt', shuffle_doc_words)

# --------- Build Vocabulary and Statistics ---------
word_freq = {}
word_doc_list = {}
for idx, doc in enumerate(shuffle_doc_words):
    words = set(doc.split())
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
        word_doc_list.setdefault(word, []).append(idx)

vocab = list(word_freq.keys())
vocab_size = len(vocab)
word_id_map = {word: idx for idx, word in enumerate(vocab)}

write_lines(f'data/corpus/{dataset}_vocab.txt', vocab)

word_doc_freq = {word: len(docs) for word, docs in word_doc_list.items()}

# --------- Label Processing ---------
labels = list({name.split('\t')[2] for name in shuffle_doc_names})
write_lines(f'data/corpus/{dataset}_labels.txt', labels)

# --------- Feature Matrices ---------
word_embeddings_dim = 300
word_vector_map = {}

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)

write_lines(f'data/{dataset}.real_train.name', shuffle_doc_names[:real_train_size])

def build_doc_feature(docs, size):
    row, col, data = [], [], []
    for i, doc in enumerate(docs):
        words = doc.split()
        doc_vec = np.zeros(word_embeddings_dim)
        for word in words:
            if word in word_vector_map:
                doc_vec += np.array(word_vector_map[word])
        doc_vec /= len(words)
        for j in range(word_embeddings_dim):
            row.append(i)
            col.append(j)
            data.append(doc_vec[j])
    return sp.csr_matrix((data, (row, col)), shape=(size, word_embeddings_dim))

x = build_doc_feature(shuffle_doc_words[:real_train_size], real_train_size)
tx = build_doc_feature(shuffle_doc_words[train_size:], test_size)

# label one-hot
label_map = {label: idx for idx, label in enumerate(labels)}
def build_labels(names):
    result = []
    for name in names:
        label = name.split('\t')[2]
        one_hot = [0] * len(labels)
        one_hot[label_map[label]] = 1
        result.append(one_hot)
    return np.array(result)

y = build_labels(shuffle_doc_names[:real_train_size])
ty = build_labels(shuffle_doc_names[train_size:])

# allx, ally (words + training documents)
word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))
row_allx, col_allx, data_allx = [], [], []

for i in range(train_size):
    words = shuffle_doc_words[i].split()
    doc_vec = np.zeros(word_embeddings_dim)
    for word in words:
        if word in word_vector_map:
            doc_vec += np.array(word_vector_map[word])
    doc_vec /= len(words)
    for j in range(word_embeddings_dim):
        row_allx.append(i)
        col_allx.append(j)
        data_allx.append(doc_vec[j])

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(train_size + i)
        col_allx.append(j)
        data_allx.append(word_vectors[i, j])

allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))
ally = np.vstack([build_labels(shuffle_doc_names[:train_size]), np.zeros((vocab_size, len(labels)))])

# --------- Graph Construction ---------
window_size = 20
windows = []

for doc in shuffle_doc_words:
    words = doc.split()
    if len(words) <= window_size:
        windows.append(words)
    else:
        for j in range(len(words) - window_size + 1):
            windows.append(words[j:j + window_size])

word_window_freq = {}
word_pair_count = {}

for window in windows:
    appeared = set()
    for word in window:
        if word not in appeared:
            word_window_freq[word] = word_window_freq.get(word, 0) + 1
            appeared.add(word)

    for i in range(len(window)):
        for j in range(i):
            word_i_id = word_id_map[window[i]]
            word_j_id = word_id_map[window[j]]
            for pair in [(word_i_id, word_j_id), (word_j_id, word_i_id)]:
                word_pair_count[pair] = word_pair_count.get(pair, 0) + 1

row, col, weight = [], [], []
num_window = len(windows)

for (i, j), count in word_pair_count.items():
    freq_i, freq_j = word_window_freq[vocab[i]], word_window_freq[vocab[j]]
    pmi = log((count / num_window) / (freq_i * freq_j / (num_window * num_window)))
    if pmi > 0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)

# --------- Document-Word tf-idf Weights ---------
for doc_id, doc in enumerate(shuffle_doc_words):
    word_count = {}
    for word in doc.split():
        word_count[word] = word_count.get(word, 0) + 1
    for word, count in word_count.items():
        word_id = word_id_map[word]
        idf = log(len(shuffle_doc_words) / word_doc_freq[word])
        if doc_id < train_size:
            row.append(doc_id)
        else:
            row.append(doc_id + vocab_size)
        col.append(train_size + word_id)
        weight.append(count * idf)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

# --------- Save Outputs ---------
outputs = {
    f"data/ind.{dataset}.x": x,
    f"data/ind.{dataset}.y": y,
    f"data/ind.{dataset}.tx": tx,
    f"data/ind.{dataset}.ty": ty,
    f"data/ind.{dataset}.allx": allx,
    f"data/ind.{dataset}.ally": ally,
    f"data/ind.{dataset}.adj": adj
}

for path, obj in outputs.items():
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

print("All data saved!")
