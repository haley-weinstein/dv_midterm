from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import nltk
import re

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        for t in word_tokenize(articles):
            t = self.wnl.lemmatize(t)
            t = [re.sub('\S*@\S*\s?', '', sent) for sent in t]

            return t


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

count_vect_1 = CountVectorizer()  # vocabulary 1
count_vect_2 = CountVectorizer(tokenizer=LemmaTokenizer())  # keeps words of 3 or more characters) # vocabulary 2

count_vect_1.fit_transform(twenty_train.data)
count_vect_2.fit_transform(twenty_train.data)
print(len(set(count_vect_1.get_feature_names())))
print(count_vect_1.get_stop_words())
print(len(count_vect_2.get_feature_names()))
