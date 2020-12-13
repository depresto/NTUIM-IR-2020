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
import pickle
from functools import reduce
from collections import Counter

def chi_square(f_obs, f_exp):
    if (len(f_obs) != len(f_exp)):
        raise Exception('Size of observed frequencies and expected frequencies is not matched')
    else:
        return reduce(lambda acc, obs: acc + (obs[1] - f_exp[obs[0]])**2 / f_exp[obs[0]], enumerate(f_obs), 0)

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

        document_token[filename] = list(set(filtered_tokens))

    return document_token

def read_training_data(document_token):
    x_train = []
    y_train = []
    training_class_ids = []
    training_doc_ids_dict = {}
    with open('./training.txt') as training_file:
        training_raw = training_file.read()

        for row in training_raw.split('\n'):
            class_id, *doc_ids = row.split(' ')[:-1]
            class_id = int(class_id)
            for doc_id in doc_ids:
                y_train.append(class_id)
                x_train.append(document_token[int(doc_id)])
            
    return x_train, y_train

class MultipleNBClassifier:
    def __init__(self):
        self.condprob = {}
        self.prior = {}
        self.C = []
        self.V = []

    def read_model(self):
        with open('./model.pickle', 'rb') as file:
            self.condprob, self.prior, self.V, self.C = pickle.load(file)

    '''
    TrainMultinomialNB(C, D)
        V <- ExtractVocabulary(D)
        N <- CountDocs(D)

        for each c in C
        do 
            Nc <- CountDocsInClass(D, c)
            prior[c] <- Nc / N
            textc <- ConcatenateTextOfAllDocsInClass(D, c)

            for each t in V
            do 
                Tct <- CountTokensOfTerm(textc, t)
            for each t in V
            do 
                condprob[t][c] <- (Tct+1) / ∑(Tct’+1)

        return V, prior, condprob
    '''
    def fit(self, x, y, k_features = float('nan'), method = 'chi_square', save = False):
        self.V = self._extract_vocabulary(x)
        N = np.unique(x).shape[0]
        self.C = np.unique(y)
        C_len = self.C.shape[0]
        
        for c in self.C:
            print('Training class', c)
            if k_features != k_features: # is nan
                selected_V = self.V
            else:
                selected_V = self._select_features(x, y, c, int(k_features / C_len), method)

            N_c = y.count(c)
            self.prior[c] = N_c / N
            text_c = self._concatenate_text_of_all_docs_in_class(x, y, c)
            
            uniq_text_c_len = len(set(text_c))
            selected_V_len = len(selected_V)
            for t in selected_V:
                T_ct = text_c.count(c)
                self.condprob[t, c] = (T_ct + 1) / (uniq_text_c_len + selected_V_len)

        if save:
            with open('./model-{}.pickle'.format(method), 'wb') as file:
                pickle.dump([self.condprob, self.prior, self.V, self.C], file)
            print('Model saved!')

        print('Training done')

    '''
    ApplyMultinomialNB(C, V, prior, condprob, d)
        W <- ExtractTokensFromDoc(V, d)
        for each c in C
        do 
            score[c] <- log prior[c]
            for each t in W
            do 
                score[c] += log condprob[t][c]

        return argmaxcscore[c]
    '''
    def predict_proba(self, W):
        score = {}
        for c in self.C:
            score[c] = math.log(self.prior[c])
            for t in W:
                if self.condprob.get((t, c)):
                    # print(math.log(self.condprob[t, c]), self.condprob[t, c], t, c)
                    score[c] += math.log(self.condprob[t, c])

        sorted_score = {k: v for k, v in sorted(score.items(), key=lambda item: item[1])}
        return sorted_score
    
    def predict(self, W):
        sorted_score = self.predict_proba(W)
        return list(sorted_score.keys())[0]
            
    def _concatenate_text_of_all_docs_in_class(self, x, y, c):
        word_list = []
        for index, cls in enumerate(y):
            if cls == c:
                word_list.extend(x[index])
        return word_list
    '''
    SelectFeatures(D, c, k)
      V <- ExtractVocabuliary(D)
      L <- []
      for each t in V
      do
          A(t,c) <- ComputeFeatureUtility(D,t,c)
          Append(L, <t, A(t,c)>)
      return FeaturesWithLargestValues(L,k)
    '''
    def _select_features(self, x, y, c, k_features, method):
        V = self._extract_vocabulary(x)
        L = {}
        for t in V:
            l = self._compute_feature_utility(x, y, t, c, method)
            if L.get(t):
                if l > L[t]:
                    L[t] = l
            else:
                L[t] = l
        sorted_L = {k: v for k, v in sorted(L.items(), key=lambda item: item[1], reverse=True)}
        return list(sorted_L.keys())[:k_features]
    
    def _compute_feature_utility(self, x, y, t, c, method = 'chi_square'):
        n_docs_on_topic_present = 0
        n_docs_on_topic_absent = 0
        n_docs_off_topic_present = 0
        n_docs_off_topic_absent = 0
        for index, doc_tokens in enumerate(x):
            if y[index] == c:
                if t in doc_tokens:
                    n_docs_on_topic_present += 1
                else:
                    n_docs_on_topic_absent += 1
            else:
                if t in doc_tokens:
                    n_docs_off_topic_present += 1
                else:
                    n_docs_off_topic_absent += 1

        n_docs_in_data = len(y)
        n_docs_on_topic = n_docs_on_topic_present + n_docs_on_topic_absent
        n_docs_off_topic = n_docs_off_topic_present + n_docs_off_topic_absent
        n_docs_present = n_docs_on_topic_present + n_docs_off_topic_present
        n_docs_absent = n_docs_on_topic_absent + n_docs_off_topic_absent
        
        e_on_topic_present = n_docs_present * n_docs_on_topic / n_docs_in_data
        e_on_topic_absent = n_docs_absent * n_docs_on_topic / n_docs_in_data
        e_off_topic_present = n_docs_present * n_docs_off_topic / n_docs_in_data
        e_off_topic_absent = n_docs_absent * n_docs_off_topic / n_docs_in_data
        
        if method == 'chi_square':
            chi = 0
            chi += (n_docs_on_topic_present - e_on_topic_present)**2 / e_on_topic_present
            chi += (n_docs_on_topic_absent - e_on_topic_absent)**2 / e_on_topic_absent
            chi += (n_docs_off_topic_present - e_off_topic_present)**2 / e_off_topic_present
            chi += (n_docs_off_topic_absent - e_off_topic_absent)**2 / e_off_topic_absent
            return chi

        if method == 'likelihood':
            p_t = (n_docs_on_topic_present + n_docs_off_topic_present) / n_docs_in_data
            p_1 = n_docs_on_topic_present / (n_docs_on_topic_present + n_docs_on_topic_absent)
            p_2 = n_docs_off_topic_present / (n_docs_off_topic_present + n_docs_off_topic_absent)

            lmbd = (p_t** n_docs_on_topic_present * (1 - p_t)** n_docs_on_topic_absent * p_t** n_docs_off_topic_present * (1 - p_t)** n_docs_off_topic_absent) / \
                (p_1** n_docs_on_topic_present * (1 - p_1)** n_docs_on_topic_absent * p_2** n_docs_off_topic_present * (1 - p_2)** n_docs_off_topic_absent)

            return -2 * math.log(lmbd)

        if method == 'MI':
            return n_docs_on_topic_present / n_docs_in_data * math.log(n_docs_in_data * n_docs_on_topic_present / n_docs_on_topic * n_docs_present or 1, 2) + \
                n_docs_off_topic_present / n_docs_in_data * math.log(n_docs_in_data * n_docs_off_topic_present / n_docs_off_topic * n_docs_present or 1, 2) + \
                n_docs_on_topic_absent / n_docs_in_data * math.log(n_docs_in_data * n_docs_on_topic_absent / n_docs_on_topic * n_docs_absent or 1, 2) + \
                n_docs_off_topic_absent / n_docs_in_data * math.log(n_docs_in_data * n_docs_off_topic_absent / n_docs_off_topic * n_docs_absent or 1, 2)
    
    def _extract_vocabulary(self, docs):
        token_list = []
        for doc in docs:
            token_list.extend(doc)
        return np.unique(token_list)
        
# Read data
document_token = read_dataset()
x_train, y_train = read_training_data(document_token)

# Initialize classifier
clf = MultipleNBClassifier()
clf.fit(x_train, y_train, k_features = 500, save = True)

# Save predict results
with open('./hw3_sam (1).csv', newline='') as csvfile:
    fieldnames = ['Id', 'Value']
    optfile = open('./hw3_sam.csv', 'w', newline='')
    writer = csv.DictWriter(optfile, fieldnames=fieldnames)
    writer.writeheader()

    rows = csv.DictReader(csvfile)
    for row in rows:
        y_pred = clf.predict(document_token[ int(row['Id']) ])
        writer.writerow({'Id': int(row['Id']), 'Value': y_pred})

    optfile.close()    