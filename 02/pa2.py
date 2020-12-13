import json
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os
import string
import csv
import math
from collections import Counter

DATA_DIR = './IRTM'
OPT_DIR = './OPT'

nltk.download('stopwords')

filename_list = []
full_filenames = os.listdir(DATA_DIR)
for full_filename in full_filenames:
    filename, extname = full_filename.split('.')
    if extname == 'txt':
        filename_list.append(int(filename))
filename_list.sort()

punct = str.maketrans('', '', string.punctuation.replace("-", ""))
def tokenize_data(content):
    # Tokenize content
    tokens = content.translate(punct).replace('\n', '').split(' ')
    # Lower case
    lower_tokens = list(map(lambda word: word.lower(), tokens))
    # Stemming using Porter's algorithm
    porter = PorterStemmer()
    stemed_tokens = list(map(lambda word: porter.stem(word), lower_tokens))
    # Stopword removal
    filtered_tokens = [
        word for word in stemed_tokens if word not in stopwords.words('english')]
    return filtered_tokens

document_frequency = Counter()
document_token_raw = []
document_token = []
document_tfs = []
for filename in filename_list:
    file = open('{}/{}.txt'.format(DATA_DIR, filename), 'r')
    content = file.read()
    file.close()

    tokens = tokenize_data(content)
    document_token_raw.append(tokens)
    filtered_tokens = filter(lambda x: x != '', tokens)
    filtered_tokens = filter(lambda x: x[0] not in string.punctuation, filtered_tokens)
    filtered_tokens = list(filter(lambda x: not str.isdigit(x[0][0]), filtered_tokens))

    word_frequencies = Counter(filtered_tokens)
    document_token.append(list(word_frequencies.keys()))
    tf = {}
    for token in filtered_tokens:
        if tf.get(token):
            tf[token] += 1
        else:
            tf[token] = 1
    document_tfs.append(tf)
    
    document_frequency.update(word_frequencies.keys())

dictionary = sorted(document_frequency.most_common(), key=lambda df: df[0])
dictionary = [(index + 1, item[0], item[1])
              for index, item in enumerate(dictionary)]
dictionary_dict = {item[1]: item for item in dictionary}

with open('./dictionary.txt', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter="\t")
    writer.writerow(['t_index', 'term', 'df'])

    for item in dictionary:
        writer.writerow(item)

n = len(filename_list)
document_tfidf = []
for index, tokens in enumerate(document_token):
    document_tf = document_tfs[index]
    total_freq = sum(document_tf.values())
    tfs = list(map(lambda item: (item[0], item[1] / total_freq), document_tf.items() ))
    dfs = [dictionary_dict.get(token) for token in tokens]
    idfs = [(df[0], math.log10(n / df[2])) for df in dfs]
    tfidfs = [(idfs[index][0], tf[1] * idfs[index][1]) for index, tf in enumerate(tfs)]
    sorted_tfidfs = sorted(tfidfs, key=lambda tfidf: tfidf[0])
    document_tfidf.append(tfidfs)
    
    with open('{}/{}.txt'.format(OPT_DIR, index + 1), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['t_index', 'tf-idf'])
        
        for tfidf in sorted_tfidfs:
            writer.writerow(tfidf)

def cosine_similarity(v1, v2):
    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
    for i in range(0, len(v1)):
        x, y = v1[i], v2[i]
        sum_xx += math.pow(x, 2)
        sum_yy += math.pow(y, 2)
        sum_xy += x * y
    try:
        return sum_xy / math.sqrt(sum_xx * sum_yy)
    except ZeroDivisionError:
        return 0

def get_document_vector(tokens, tfidfs):
    vector = []
    for token in tokens:
        dictionary_item = dictionary_dict.get(token)
        if dictionary_item is None:
            vector.append(0)
        else:
            dictionary_index = dictionary_item[0]
            tfidf = next(filter(lambda tfidf: tfidf[0] == dictionary_index, tfidfs), None)
            vector.append(tfidf[1])
    return vector

vec_doc1 = get_document_vector(document_token_raw[0], document_tfidf[0])
vec_doc2 = get_document_vector(document_token_raw[1], document_tfidf[1])
print(cosine_similarity(vec_doc1, vec_doc2))