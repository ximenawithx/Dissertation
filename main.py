import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
# %matplotlib inline

# Loading and reading file from a csv extension
file = pd.read_csv('labeled_data.csv')
file.describe()
file.columns

# Creating a histogram with the data
file['class'].hist()
tweets = file.tweet
print("file", file)

print("tweets = ", tweets)

# Feature Generation
stopwords = stopwords = (stopwords.words("english"))
print("stopwords")
print(type(stopwords))
print(stopwords [30-100])

exclusions = ["#ff", "ff", "rt"]
stopwords.extend(exclusions)

stemmer = PorterStemmer()

#     Preprocessing data
def preprocess(text_string):
    space = '\s+'
    url_regex = ('http[s]?: // (?:[a-zA-Z]|[0-9][$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space, '', text_string)
    parsed_text = re.sub(url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text
    print("parsed text = ", parsed_text)

def tokenize (tweet):
    # removing punctuation, unnecessary whitespace, sets all to lowercase
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    print("tokens", tokens)
    return tokens

def basic_tokenize(tweet):
    # same as tokenize method but without the stemming
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    print("tweet split = ", tweet.split())
    return tweet.split()

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.501
)
#  Construction of tfidf matrix
tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i: idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF indices

# Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tags_list = [x[1] for x in tags]
    tag_str = " ".join(tags_list)
    tweet_tags.append(tag_str)

#Use of TFIDF vectorizer to get token matrix for POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None, #we do better when keeping stopwords
    use_idf=False,
    smooth_idf=False,
    norm=None, #Applies L2 norm smoothing
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.501,
)

# Construction of POS TF matrix and get vocab dict
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

# Getting other features
sentiment_analyzer = VS()

def count_twitter_objs(text_string):
# Accepts a text string and replaces:
# 1) urls with URLHERE
# 2) lots of whitespace with one instance
# 3) mentions with MENTIONHERE
# 4) Hashtags with HASHTAGHERE
#  This allows to get standarized counts of urls and mentions wihout caring about specific people mentioned
# Returns counts of urls, mentions, and hashtags
    space_pattern='\s+'
    url_regex=('https[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex='@[\w\-]+'
    hashtag_regex='#[\w\-]+'
    parsed_text=re.sub(space_pattern,' ',text_string)
    parsed_text= re.sub(url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    #     This function takes a string and returns a list of features.
    #  With these we get Sentiment scores, Text and Redability scores, as well as Twitter specifc features
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocess(tweet) #gets text only
    syllables = textstat.syllable_count(words) # count syllables in words
    num_chars= sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+ 0.001), 4)
    num_unique_terms = len(set(words.split()))

    #Modified FK grade, avg per sentence = num words/1
    FKRA = round(float(0.39 *float(num_words)/1.0) + float(11.8 *avg_syl) - 15.59, 1)
    #Modified FRE score, sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0)- (84.6*float(avg_syl)),2)

    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    retweet = 0
    if "x" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1], twitter_objs[0], retweet]
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", "vader compound", \
                        "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

feats = get_feature_array(tweets)

# Join all together
M= np.concatenate([tfidf, pos, feats], axis=1)
M.shape

# List of variable names
variables = ['']*len(vocab)
for k,v in pos_vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_variables[v] = k
feature_names = variables + pos_variables + other_features_names



# # dict = {'one': 1, 'two': 2}
# file = open('labeled_data.txt', 'w')
# pickle.dump(file)
#
# print("labeled data: ", file)
#
# file.close()

# data = pickle.load(open("labeled_data.txt", 'rb'))
# tweets = data.text

# print(data)