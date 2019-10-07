"""
Author: Hayle
Description: Preprocess some shit idk
"""
import nltk
import os

PATH_TO_NEWS_GROUP = 'C:\\Users\\l\\Desktop\\Data Vis\\midterm\\20_newsgroups'

class TotalCorp(object):
    def __init__(self, path_to_news_group=PATH_TO_NEWS_GROUP, folder_name='alt.atheism', b='51174'):
        self.p = path_to_news_group
        self.folder = folder_name
        self.corpus = []
        self.word_freq_dict = {}
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') # drops everything except alphanumeric characters
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
            tokens = self.tokenizer.tokenize(self.corpus[idx])

            for token in tokens:
                if token not in freq_dict.keys():
                    freq_dict[token] = 1
                else:
                    freq_dict[token] += 1
                total_words += 1

        # print("example of processed token:{}".format(tokens))  # prints example of word tokens can be taken out later
        return freq_dict, total_words

aethism = TotalCorp()
aethism.create_corp()
