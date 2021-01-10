# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import os
import string
import math
from tqdm import tqdm


# %%
k_list = [8, 13, 20]
DATA_DIR = './IRTM'
OPT_DIR = './OPT'


# %%
def tokenize_data(content):
    punct = str.maketrans('', '', string.punctuation.replace("-", ""))
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

def read_dataset():
    filename_list = []
    full_filenames = os.listdir('./IRTM')
    for full_filename in full_filenames:
        filename, extname = full_filename.split('.')
        if extname == 'txt':
            filename_list.append(int(filename))

    document_token = {}
    for filename in filename_list:
        file = open('{}/{}.txt'.format('./IRTM', filename), 'r')
        content = file.read()
        file.close()

        tokens = tokenize_data(content)

        filtered_tokens = filter(lambda x: x != '', tokens)
        filtered_tokens = filter(lambda x: x[0] not in string.punctuation, filtered_tokens)
        filtered_tokens = list(filter(lambda x: not str.isdigit(x[0][0]), filtered_tokens))

        document_token[filename] = filtered_tokens

    return document_token

def cosine_similarity(d1, d2):
    return d1 @ d2 / (((d1 @ d1) ** 0.5) * ((d2 @ d2) ** 0.5))


# %%
document_token = read_dataset()


# %%
document_token_count = {}
for filename, tokens in document_token.items():
    document_token_count[filename] = Counter(tokens)

document_token_frequency = {}
for filename, token_counter in document_token_count.items():
    token_length = sum(token_counter.values())
    frequency = {}
    for token, count in token_counter.items():
        frequency[token] = count / token_length
    document_token_frequency[filename] = frequency

global_token = []
for token_counter in document_token_count.values():
    global_token.extend(token_counter.keys())
global_token_occurence = Counter(global_token)

inverse_document_frequency = {}
for filename, token_counter in document_token_count.items():
    inverse_frequency = {}
    for token in token_counter.keys():
        token_occurence = global_token_occurence[token]
        inverse_frequency[token] = math.log10(len(document_token_count) / token_occurence)
    inverse_document_frequency[filename] = inverse_frequency

document_tf_idf = {}
for document, token_frequency in document_token_frequency.items():
    tf_idf = {}
    for token, frequency in token_frequency.items():
        tf_idf[token] = frequency * inverse_document_frequency[document][token]
    document_tf_idf[document] = tf_idf


# %%
dictionary = sorted(global_token_occurence.items(), key=lambda item: item[0])
dictionary_token = list(map(lambda item: item[0], dictionary))


# %%
document_tf_idf_list = []
for tf_idfs in document_tf_idf.values():
    tf_idf_list = [0] * len(dictionary_token)
    for token, tf_idf in tf_idfs.items():
        tf_idf_list[dictionary_token.index(token)] = tf_idf
    document_tf_idf_list.append(tf_idf_list)

document_tf_idf_arr = np.array(document_tf_idf_list)


# %%
def simple_hac(documents):
    N = len(documents)
    C = {}
    I = [False] * N
    pbar_progress = tqdm(range(N), desc="Load similarity", position=0, leave=True)
    for n in pbar_progress:
        for i in range(N):
            C[n, i] = cosine_similarity(documents[n], documents[i])
        I[n] = True
    A = []

    pbar_merge = tqdm(range(N - 1), desc="Merge clusters", position=0, leave=True)
    for k in pbar_merge:
        max_sum = {
            'i': None,
            'm': None,
            'cosine_similarity': 0
        }
        for i in range(N):
            if I[i] is False:
                continue
            for m in range(N):
                if I[m] is False or i == m:
                    continue
                if C[i, m] > max_sum['cosine_similarity']:
                    max_sum['cosine_similarity'] = C[i, m]
                    max_sum['i'] = i
                    max_sum['m'] = m

        i = max_sum['i']
        m = max_sum['m']
        sim = cosine_similarity(documents[i], documents[m])
        pbar_merge.set_description("Similarity: {}. Merge {}, {}".format(sim, i, m))
        A.append({ 'i': i, 'm': m, 'cosine_similarity': sim })

        for j in range(N):
            # Complete link
            C[i, j] = min(cosine_similarity(documents[i], documents[j]), cosine_similarity(documents[m], documents[i]))
            C[j, i] = min(cosine_similarity(documents[j], documents[i]), cosine_similarity(documents[m], documents[j]))
        I[m] = False

    return C, A



# %%
C, A = simple_hac(document_tf_idf_arr)


# %%
k_clusters = []
for k in k_list:
    clusters = []
    for i in range(len(document_tf_idf_arr)):
        clusters.append([i + 1])
    clusters_len = len(clusters)

    for a in A:
        clusters[a['i']].extend(clusters[a['m']])
        clusters[a['m']] = None
        clusters_len -= 1

        if clusters_len == k:
            clusters = list(filter(lambda cluster: cluster is not None, clusters))
            k_clusters.append(clusters)
            break


# %%
for index, k in enumerate(k_list):
    with open('{}/{}.txt'.format(OPT_DIR, k), 'w') as opt_file:
        for doc_ids in k_clusters[index]:
            for doc_id in sorted(doc_ids):
                opt_file.write("{}\n".format(doc_id))
            opt_file.write("\n")


