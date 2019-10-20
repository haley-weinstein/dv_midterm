"""
Author: Hayle
Description: Preprocess some stuff idk
I put EFFORT in but guess we will j b sheeple who use tutorials
"""
import nltk
import heapq
import matplotlib.pyplot as plt
import matplotlib.cm as c
import os
import math
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import json
import zlib
import base64
import pickle

ZIPJSON_KEY = 'base64(zip(o))'


def json_zip(j, fp):
    j = {
        ZIPJSON_KEY: base64.b64encode(
            zlib.compress(
                json.dump((j).encode('utf-8'), fp)
            )
        ).decode('ascii')
    }

    return j


NUMBER_OF_WORDS = 200
PATH_TO_NEWS_GROUP = 'C:\\Users\\Zacha\\Desktop\\20_newsgroups'


class TotalCorp(object):
    def __init__(self, path_to_news_group=PATH_TO_NEWS_GROUP, folder_name='alt.atheism', b='51060', lemma=False):
        self.p = path_to_news_group
        self.folder = folder_name
        self.corpus = []
        self.word_freq_dict = {}
        self.lemmatizer = WordNetLemmatizer()
        self.lemma = lemma
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # drops everything except alphanumeric characters
        self.total_words = 0
        self.break_at = b
        self.word_dict = {}

    def create_corp(self):
        total_documents = len(os.listdir(os.path.join(self.p, self.folder)))
        for idx, f in enumerate(os.listdir(os.path.join(self.p, self.folder))):
            print("{}_{}".format(idx, total_documents))
            a = open(os.path.join(self.p, self.folder, f), 'r')
            self.corpus = self.corpus + (nltk.sent_tokenize(a.read()))
            self.word_freq_dict[f] = {}
            self.word_dict[f] = []
            self.word_freq_dict[f]['frequencies'], self.word_freq_dict[f]['total_words'], self.word_freq_dict[
                f]['comma_seperated_words'], self.word_freq_dict[f]['sentence_vectors'] = self.find_frequency()
            # _, _, _, self.word_freq_dict[f]['sentence)vectors'] = self.find_frequency()
            if str(f) == self.break_at:  # it takes a really long time to run so I j did this
                break

    def save(self):
        print("SAVING CORP")
        if self.lemma:
            pickle.dump(self.word_freq_dict, open("{}_{}.p".format("NO_LEMMA", self.folder), "wb"))
        else:
            pickle.dump(self.word_freq_dict, open("{}_{}.p".format("NO_LEMMA", self.folder), "wb"))

    def find_frequency(self):
        freq_dict = {}
        word_dict = []
        total_words = 0
        sentence_vecs = []
        for idx, cor in enumerate(self.corpus):
            self.corpus[idx] = cor.lower().replace("_", "")
            tokens = self.tokenizer.tokenize(self.corpus[idx])
            tokens = [re.sub('\S*@\S*\s?', '', sent) for sent in tokens]
            tokens = [re.sub('\s+', ' ', sent) for sent in tokens]
            tokens = [re.sub("\'", "", sent) for sent in tokens]

            if self.lemma:
                stop_words = set(stopwords.words('english'))
                tokens = [w for w in tokens if not w in stop_words]
            tokens_ = []
            for token in tokens:
                if self.lemma:
                    token = self.lemmatizer.lemmatize(token)

                if token not in freq_dict.keys():
                    freq_dict[token] = 1
                else:
                    freq_dict[token] += 1
                total_words += 1
                word_dict.append(token)
                tokens_.append(token)
            sentence_vecs.append(tokens_)

        # print("example of processed token:{}".format(tokens))  # prints example of word tokens can be taken out later
        return freq_dict, total_words, word_dict, sentence_vecs


class BagOfWords(object):
    def __init__(self, corpus, word_freq_dict, words=NUMBER_OF_WORDS):
        self.words = words
        self.corpus = corpus
        self.wordfreq = word_freq_dict
        self.sentence_vecs = []
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # drops everything except alphanumeric characters

    def create_sentence_vectors(self):
        if self.wordfreq == {}:
            raise ValueError("Create the frequency dictionary in the corpus class pls")
        most_freq = heapq.nlargest(NUMBER_OF_WORDS, self.wordfreq, key=self.wordfreq.get)

        for sentence in self.corpus:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sent_vec = []
            for token in most_freq:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            self.sentence_vecs.append(sent_vec)

    def plot_vectors(self, c_map='jet'):
        """this is hella ugly i will try to make it better eventually lol """
        cmap = getattr(c, c_map)
        plt.imshow(self.sentence_vecs, cmap=cmap)
        plt.show()


class TFIDF(object):
    def __init__(self, corpus, wordfreq):
        self.corpus = corpus
        self.wordfreq = wordfreq
        self.tf_scores = {}
        self.tfidf_scores = {}
        self.word_instances = {}

    def compute_tf(self):
        """Frequency of term in doc/ total number of terms in the doc"""
        for k in self.wordfreq.keys():
            self.tf_scores[k] = {}
            for k2 in self.wordfreq[k]['frequencies'].keys():  # k2 = word name

                word_count = self.wordfreq[k]['frequencies'][k2]
                self.tf_scores[k][k2] = word_count / self.wordfreq[k]['total_words']

    def compute_tfidf(self):
        """ln(total number of docs/ number of docs with term in it) """
        self.compute_tf()
        total_docs = len(self.wordfreq.keys())
        for k in self.wordfreq.keys():
            self.tfidf_scores[k] = {}
            for k2 in self.wordfreq[k]['frequencies'].keys():
                if k2 in self.word_instances.keys():
                    self.tfidf_scores[k][k2] = self.tf_scores[k][k2] * math.log(total_docs / self.word_instances[k2])
                else:
                    self.tfidf_scores[k][k2] = self.tf_scores[k][k2] * math.log(
                        total_docs / self.find_total_occurrences(k2))

    def find_total_occurrences(self, word):
        instance = 0
        for k in self.wordfreq.keys():
            if word in self.wordfreq[k]['frequencies'].keys():
                instance += 1

        self.word_instances[word] = instance
        return instance


aethism = TotalCorp()
aethism.create_corp()

# bag_aethism = BagOfWords(aethism.corpus, aethism.word_freq_dict['comma_seperated_words'])
# bag_aethism.create_sentence_vectors()
"""
for c in os.listdir(PATH_TO_NEWS_GROUP):
    print(c)
    corp = TotalCorp(lemma=False, folder_name=c, b='')
    corp.create_corp()
    corp.save()


wordcloud = WordCloud()
wordcloud.generate_from_frequencies(aethism.word_freq_dict['51060']['frequencies'])
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
bag_aethism = BagOfWords(aethism.corpus, aethism.word_freq_dict['51060']['frequencies'])
bag_aethism.create_sentence_vectors()

TFIDF_aethism = TFIDF(aethism.corpus, aethism.word_freq_dict)
TFIDF_aethism.compute_tfidf()
"""
