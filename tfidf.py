"""
Author: Hayle
Description: Create a TFIDF model
"""
import math
from pre_process import aethism

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

TFIDF_aethism = TFIDF(aethism.corpus, aethism.word_freq_dict)
TFIDF_aethism.compute_tfidf()