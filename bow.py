"""
Author: Hayle
Description: Create a Bag-of-words model
"""
import nltk
import heapq
import matplotlib.pyplot as plt
import matplotlib.cm as c

from pre_process import aethism

NUMBER_OF_WORDS = 200

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


bag_aethism = BagOfWords(aethism.corpus, aethism.word_freq_dict['51060']['frequencies'])
bag_aethism.create_sentence_vectors()
#bag_aethism.plot_vectors()