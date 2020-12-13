from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from functools import reduce
import nltk
import json
import re

# Read file
file = open('28.txt', 'r')
content = file.read()
file.close()

# Tokenize content
tokens = re.sub(r'\n|,|\.|\'', '', content).split(' ')
# Lower case
lower_tokens = list(map(lambda word: word.lower(), tokens))
# Stemming using Porter's algorithm
porter = PorterStemmer()
stemed_tokens = list(map(lambda word: porter.stem(word), lower_tokens))
# Stopword removal
nltk.download('stopwords')
filtered_tokens = [word for word in stemed_tokens if word not in stopwords.words('english')]
# Save as txt file
result_file = open('result.txt', 'w')
result_file.write(json.dumps(filtered_tokens))
result_file.close()