"""
Author: Hayle
Description: Preprocess some shit idk
"""
import nltk
import heapq
import matplotlib.pyplot as plt
import matplotlib.cm as c
import os
import math

NUMBER_OF_WORDS = 200
PATH_TO_NEWS_GROUP = 'C:\\Users\\haleyweinstein\\Documents\\20_newsgroups'


class TotalCorp(object):
    def __init__(self, path_to_news_group=PATH_TO_NEWS_GROUP, folder_name='alt.atheism', b='51174'):
        self.p = path_to_news_group
        self.folder = folder_name
        self.corpus = []
        self.word_freq_dict = {}
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # drops everything except alphanumeric characters
        self.total_words = 0
        self.break_at = b

    def create_corp(self):
        for f in os.listdir(os.path.join(self.p, self.folder)):
            a = open(os.path.join(self.p, self.folder, f), 'r')
            self.corpus = self.corpus + (nltk.sent_tokenize(a.read()))
            self.word_freq_dict[f] = {}
            self.word_freq_dict[f]['frequencies'], self.word_freq_dict[f]['total_words'] = self.find_frequency()
            if str(f) == self.break_at:  # it takes a really fucking long time to run so I j did this
                break

    def find_frequency(self):
        freq_dict = {}
        total_words = 0
        for idx, cor in enumerate(self.corpus):
            self.corpus[idx] = cor.lower().replace("_", "")
            tokens = self.tokenizer.tokenize(cor.lower().replace("_", ""))

            for token in tokens:
                if token not in freq_dict.keys():
                    freq_dict[token] = 1
                else:
                    freq_dict[token] += 1
                total_words += 1

        # print("example of processed token:{}".format(tokens))  # prints example of word tokens can be taken out later
        return freq_dict, total_words


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
        "Frequency of term in doc/ total number of terms in the doc"
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
                        total_docs / self.find_total_occurances(k2))

    def find_total_occurances(self, word):
        instance = 1
        for k in self.wordfreq.keys():
            if word in self.wordfreq[k]['frequencies'].keys():
                instance += 1

        self.word_instances[word] = instance
        return instance


aethism = TotalCorp()
aethism.create_corp()

bag_aethism = BagOfWords(aethism.corpus, aethism.word_freq_dict['51060']['frequencies'])
bag_aethism.create_sentence_vectors()
bag_aethism.plot_vectors()

TFIDF_aethism = TFIDF(aethism.corpus, aethism.word_freq_dict)
TFIDF_aethism.compute_tfidf()
