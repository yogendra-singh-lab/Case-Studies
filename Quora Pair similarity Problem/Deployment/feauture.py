# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:31:17 2021

@author: ogunniran siji
"""


import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')
import os, gc, re, distance
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def preprocess(q):
  # Firstly, we convert to lowercase and remove trailing and leading spaces
  q = str(q).lower().strip()

  # Replace certain special characters with their string equivalents
  q = q.replace('%', ' percent')
  q = q.replace('$', ' dollar ')
  q = q.replace('₹', ' rupee ')
  q = q.replace('€', ' euro ')
  q = q.replace('@', ' at ')

  # The pattern '[math]' appears around 900 times in the whole dataset.
  q = q.replace('[math]', '')

  # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
  q = q.replace(',000,000,000 ', 'b ')
  q = q.replace(',000,000 ', 'm ')
  q = q.replace(',000 ', 'k ')
  q = re.sub(r'([0-9]+)000000000', r'\1b', q)
  q = re.sub(r'([0-9]+)000000', r'\1m', q)
  q = re.sub(r'([0-9]+)000', r'\1k', q)

  # Decontracting words
  # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
  # https://stackoverflow.com/a/19794953
  contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
  }

  q_decontracted = []

  for word in q.split():
    if word in contractions:
      word = contractions[word]
  
    q_decontracted.append(word)

  q = ' '.join(q_decontracted)
  q = q.replace("'ve", " have")
  q = q.replace("n't", " not")
  q = q.replace("'re", " are")
  q = q.replace("'ll", " will")

  # Removing HTML tags
  q = BeautifulSoup(q)
  q = q.get_text()

  # Remove punctuations
  pattern = re.compile('\W')
  q = re.sub(pattern, ' ', q).strip()

  return q

"""## Extracting features
We will extract following features:
- **Token features**
  1. **q1_len**: Number of characters in question 1
  1. **q2_len**: Number of characters in question 2
  1. **q1_words**: Number of words in question 1
  1. **q2_words**: Number of words in question 2
  1. **words_total**: Sum of **q1_words** and **q2_words**
  1. **words_common**: Number of words which occur in question 1 and two, reapeated occurances are not counted
  1. **words_shared**: Fraction of **words_common** to **words_total**
  1. **cwc_min**: This is the ratio of the number of common words to the length of the smaller question
  1. **cwc_max**: This is the ratio of the number of common words to the length of the larger question
  1. **csc_min**: This is the ratio of the number of common stop words to the smaller stop word count among the two questions
  1. **csc_max**: This is the ratio of the number of common stop words to the larger stop word count among the two questions
  1. **ctc_min**: This is the ratio of the number of common tokens to the smaller token count among the two questions
  1. **ctc_max**: This is the ratio of the number of common tokens to the larger token count among the two questions
  1. **last_word_eq**: 1 if the last word in the two questions is same, 0 otherwise
  1. **first_word_eq**: 1 if the first word in the two questions is same, 0 otherwise
  1. **num_common_adj**: This is the number of common adjectives in question1 and question2
  1. **num_common_prn**: This is the number of common proper nouns in question1 and question2
  1. **num_common_n**: This is the number of nouns (non-proper) common in question1 and question2
- **Fuzzy features**
  1. **fuzz_ratio**: fuzz_ratio score from fuzzywuzzy
  1. **fuzz_partial_ratio**: fuzz_partial_ratio from fuzzywuzzy
  1. **token_sort_ratio**: token_sort_ratio from fuzzywuzzy
  1. **token_set_ratio**: token_set_ratio from fuzzywuzzy
- **Length features**
  1. **mean_len**: Mean of the length of the two questions (number of words)
  1. **abs_len_diff**: Absolute difference between the length of the two questions (number of words)
  1. **longest_substr_ratio**: Ratio of the length of the longest substring among the two questions to the length of the smaller question
### Defining functions to extract features
"""

# Receives question1 and question2 from one row in DataFrame
# Computes token features, removes stopwords and performs stemming
# Returns an array of shape (num_features,)
def get_token_features(q1, q2):
  # Safe div to avoid division by 0 exception
  safe_div = 0.0001

  # Getting NLTK  stop words set
  stop_words = stopwords.words('english')
  
  # Adding these after word cloud inspection
  stop_words.append('difference')
  stop_words.append('different')
  stop_words.append('best')

  # Initializing stemmer
  stemmer = PorterStemmer()

  # Initializing feature array
  token_features = [0.0] * 18

  # Tokenizing
  q1 = q1.split()
  q2 = q2.split()

  # Stop words in q1 and q2
  q1_stops = set([word for word in q1 if word in stop_words])
  q2_stops = set([word for word in q2 if word in stop_words])
  common_stops = q1_stops & q2_stops

  # Removing stop words
  q1 = [word for word in q1 if word not in stop_words]
  q2 = [word for word in q2 if word not in stop_words]

  # Stem
  # Is redundant but this design change was made much later and 
  # I don't feel like changing the entire function for it.
  # For now, computationally inefficient though it may be, it will do.
  q1_stemmed = ' '.join([word for word in q1])
  q2_stemmed = ' '.join([word for word in q2])

  if len(q1) == 0 or len(q2) == 0:
    return (token_features, q1_stemmed, q2_stemmed)

  # PoS features
  # Uses off the shelf NLTK tag set

  q1_tagged = nltk.pos_tag(q1)
  q2_tagged = nltk.pos_tag(q2)

  # We are looking for:
  # 1) JJ/JJR/JJS: Adjectives
  # 2) NNP/NNPS: Proper nouns
  # 3) NN/NNS: Nouns (non-proper)

  q1_adj = set()
  q2_adj = set()
  q1_prn = set()
  q2_prn = set()
  q1_n = set()
  q2_n = set()

  # Compute question1 PoS features
  for word in q1_tagged:
    if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
      q1_adj.add(word[0])
    elif word[1] == 'NNP' or word[1] == 'NNPS':
      q1_prn.add(word[0])
    elif word[1] == 'NN' or word[1] == 'NNS':
      q1_n.add(word[0])

  # Compute question2 PoS features
  for word in q2_tagged:
    if word[1] == 'JJ' or word[1] == 'JJR' or word[1] == 'JJS':
      q2_adj.add(word[0])
    elif word[1] == 'NNP' or word[1] == 'NNPS':
      q2_prn.add(word[0])
    elif word[1] == 'NN' or word[1] == 'NNS':
      q2_n.add(word[0])
      
  # num_common_adj
  token_features[15] = len(q1_adj & q2_adj)

  # num_common_prn
  token_features[16] = len(q1_prn & q2_prn)

  # num_common_n
  token_features[17] = len(q1_n & q2_n)

  # We do this here because converting to set looses order of words
  # last_word_eq
  token_features[13] = int(q1[-1] == q2[-1])

  # first_word_eq
  token_features[14] = int(q1[0] == q2[0])

  # Now we convert the questions into sets, this looses order but removes duplicate words
  q1 = set(q1)
  q2 = set(q2)
  common_tokens = q1 & q2

  # Sets are still iterables, order of words won't change the number of characters
  # q1_len
  token_features[0] = len(q1_stemmed) * 1.0

  # q2_len
  token_features[1] = len(q2_stemmed) * 1.0

  # q1_words
  token_features[2] = len(q1) * 1.0

  # q2_words
  token_features[3] = len(q2) * 1.0

  # words_total
  token_features[4] = token_features[2] + token_features[3]

  # Common words
  q1_words = set(q1)
  q2_words = set(q2)
  common_words = q1_words & q2_words
  
  # words_common
  token_features[5] = len(common_words) * 1.0

  # words_shared
  token_features[6] = token_features[5] / (token_features[4] + safe_div)

  # cwc_min
  token_features[7] = token_features[5] / (min(token_features[2], token_features[3]) + safe_div)

  # cwc_max
  token_features[8] = token_features[5] / (max(token_features[2], token_features[3]) + safe_div)

  # csc_min
  token_features[9] = (len(common_stops) * 1.0) / (min(len(q1_stops), len(q2_stops)) + safe_div)

  # csc_max
  token_features[10] = (len(common_stops) * 1.0) / (max(len(q1_stops), len(q2_stops)) + safe_div)

  # ctc_min
  token_features[11] = (len(common_tokens) * 1.0) / (min(len(q1), len(q2)) + safe_div)

  # ctc_max
  token_features[12] = (len(common_tokens) * 1.0) / (max(len(q1), len(q2)) + safe_div) 

  return (token_features, q1_stemmed, q2_stemmed)


# Computes fuzzy features
# Returns an array of shape (n_features,)
def get_fuzzy_features(q1, q2):
  # Initilzing feature array
  fuzzy_features = [0.0] * 4

  # fuzz_ratio
  fuzzy_features[0] = fuzz.QRatio(q1, q2)

  # fuzz_partial_ratio
  fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

  # token_sort_ratio
  fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

  # token_set_ratio
  fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

  return fuzzy_features


# Computes length features
# Returns an array of shape (n_features,)
def get_length_features(q1, q2):
  # Safe div to avoid division by 0 exception
  safe_div = 0.0001

  # Initialzing feature array
  length_features = [0.0] * 3

  q1_list = q1.strip(' ')
  q2_list = q2.strip(' ')

  # mean_len
  length_features[0] = (len(q1_list) + len(q2_list)) / 2

  # abs_len_diff
  length_features[1] = abs(len(q1_list) - len(q2_list))

  # Get substring length
  substr_len = distance.lcsubstrings(q1, q2, positions=True)[0]

  # longest_substr_ratio
  if substr_len == 0:
    length_features[2] = 0
  else:
    length_features[2] = substr_len / (min(len(q1_list), len(q2_list)) + safe_div)

  return length_features


# Receives data set and performs cleaning, feature extractions
# Transforms data set by adding feature columns
# Returns transformed DataFrame
def extract_features(df_train):
  # First, lets call the preprocess function on question1 and question2
  df_train['Marking Scheme'] = df_train['Marking Scheme'].apply(preprocess)
  df_train['Student Response'] = df_train['Student Response'].apply(preprocess)

  # Get token features, token_features is an array of shape (n_rows, data)
  # where data is a tuple of containing (n_features, q1_stemmed, q2_stemmed)
  # token_features, q1_stemmed, q2_stemmed = data.apply(lambda x: get_token_features(x['question1'], x['question2']), axis=1)
  token_features = df_train.apply(lambda x: get_token_features(x['Marking Scheme'], x['Student Response']), axis=1)
  
  q1_stemmed = list(map(lambda x: x[1], token_features))
  q2_stemmed = list(map(lambda x: x[2], token_features))
  token_features = list(map(lambda x: x[0], token_features))

  df_train['Marking Scheme'] = q1_stemmed
  df_train['Student Response'] = q2_stemmed

  # Creating new feature columns for token features
  df_train['q1_len'] = list(map(lambda x: x[0], token_features))
  df_train['q2_len'] = list(map(lambda x: x[1], token_features))
  df_train['q1_words'] = list(map(lambda x: x[2], token_features))
  df_train['q2_words'] = list(map(lambda x: x[3], token_features))
  df_train['words_total'] = list(map(lambda x: x[4], token_features))
  df_train['words_common'] = list(map(lambda x: x[5], token_features))
  df_train['words_shared'] = list(map(lambda x: x[6], token_features)) 
  df_train['cwc_min'] = list(map(lambda x: x[7], token_features))
  df_train['cwc_max'] = list(map(lambda x: x[8], token_features))
  df_train['csc_min'] = list(map(lambda x: x[9], token_features))
  df_train['csc_max'] = list(map(lambda x: x[10], token_features))
  df_train['ctc_min'] = list(map(lambda x: x[11], token_features))
  df_train['ctc_max'] = list(map(lambda x: x[12], token_features))
  df_train['last_word_eq'] = list(map(lambda x: x[13], token_features))
  df_train['first_word_eq'] = list(map(lambda x: x[14], token_features))
  df_train['num_common_adj'] = list(map(lambda x: x[15], token_features))
  df_train['num_common_prn'] = list(map(lambda x: x[16], token_features))
  df_train['num_common_n'] = list(map(lambda x: x[17], token_features))

  # Get fuzzy features, fuzzy_features is an array of shape (n_rows, n_features)
  fuzzy_features = df_train.apply(lambda x: get_fuzzy_features(x['Marking Scheme'], x['Student Response']), axis=1)

  # Creating new feature columns for fuzzy features
  df_train['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
  df_train['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
  df_train['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
  df_train['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

  # Get length features, length_features is an array of shape (n_rows, n_features)
  length_features = df_train.apply(lambda x: get_length_features(x['Marking Scheme'], x['Student Response']), axis=1)

  # Creating new feature columns for length features
  df_train['mean_len'] = list(map(lambda x: x[0], length_features))
  df_train['abs_len_diff'] = list(map(lambda x: x[1], length_features))
  df_train['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

  return df_train
