import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')
import os, gc, re, distance
from sklearn.feature_extraction.text import TfidfVectorizer

def text_preprocess(x):
    porter = PorterStemmer()
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x) #e.g. replace 12000000 with 12m
    x = re.sub(r"([0-9]+)000", r"\1k", x) #e.g. replace 52000 with 52k
    x = re.sub(r'<.*?>', '',x) # removes the htmltags: https://stackoverflow.com/a/12982689
    # stemming the words
    x = porter.stem(x)
    return x

# Basic Feature Extraction
def basic_feature_extraction(row):
    x = [0]*9

    q1_len = len(row['question1'])
    q2_len = len(row['question2'])
    q1_n_word = len(row['question1'].split(" "))
    q2_n_word = len(row['question2'].split(" "))

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))

    q1 = row['question1'].lower().replace('?','').replace('.','').replace('!','')
    q2 = row['question2'].lower().replace('?','').replace('.','').replace('!','')
    row_words = list(set(q1.split(' ') + q2.split(' '))) #unique words in both sentences
    q1_vec, q2_vec = np.zeros((1, len(row_words))), np.zeros((1, len(row_words)))

    for word in q1.split(' '):
        q1_vec[0][row_words.index(word)] += 1
    for word in q2.split(' '):
        q2_vec[0][row_words.index(word)] += 1

    x[0], x[1] = q1_len, q2_len
    x[2], x[3] = q1_n_word, q2_n_word
    x[4] = 1.0 * len(w1 & w2)
    x[5] = 1.0 * (len(w1) + len(w2))
    x[6] = 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    x[7] = np.linalg.norm(q1_vec-q2_vec) ## to find eucldian distance:https://stackoverflow.com/a/1401828
    x[8] = (1 - np.matmul(q1_vec, q2_vec.T)/(np.linalg.norm(q1_vec) * np.linalg.norm(q2_vec)))[0][0]
    return x

# To get the results in 4 decemal points
SAFE_DIV = 0.0001

try:
    STOP_WORDS = stopwords.words("english")
except:
    import nltk
    nltk.download('stopwords')
    STOP_WORDS = stopwords.words("english")

def get_token_features(q1, q2):
    token_features = [0.0]*10
    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2

    return token_features

# get the Longest Common sub string
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    # print('1. Extracting Basic Features...')
    basic_features = df.apply(basic_feature_extraction, axis = 1)
    basic_features_names = ['q1len', 'q2len', 'q1_n_word', 'q2_n_word', 'word_Common', 'word_total', 'word_share', 'dist_eucl', 'dist_cosine']
    for i, name in enumerate(basic_features_names):
        df[name] = list(map(lambda x: x[i], basic_features))

    # preprocessing each question
    # print('2. Extracting Advance Features\n', '2.1 Text preprocessing')
    df["question1"] = df["question1"].apply(text_preprocess)
    df["question2"] = df["question2"].apply(text_preprocess)

    # Merging Features with dataset
    # print("2.2 Extracting token features...")
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    new_columns = ["cwc_min", "cwc_max", "csc_min" ,"csc_max" ,"ctc_min","ctc_max","last_word_eq" ,"first_word_eq" ,"abs_len_diff","mean_len"]
    for i, name in enumerate(new_columns):
       df[name] = list(map(lambda x: x[i], token_features))

    #Computing Fuzzy Features and Merging with Dataset
    # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy
    # print("2.3 Extracting Fuzzy Features...")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

    return df
