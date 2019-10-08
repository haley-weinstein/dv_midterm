"""
Author: Hayle
Description: Preprocess some shit idk
"""
import nltk
import os

PATH_TO_NEWS_GROUP = 'C:\\Users\\l\\Desktop\\Data Vis\\midterm\\20_newsgroups'

class TotalCorp(object):
    def __init__(self, path_to_news_group=PATH_TO_NEWS_GROUP, folder_names={'alt.atheism'}, b='51174'):
        self.p = path_to_news_group
        self.folders = folder_names
        self.corpus = []
        self.word_freq_dict = {}
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') # drops everything except alphanumeric characters
        self.total_words = 0
        self.break_at = b
        self.word_dict = {}

    def create_corp(self):
        for folder in os.listdir(self.p):
            if folder not in self.folders and len(self.folders) > 0:
                continue
            for file in os.listdir(os.path.join(self.p, folder)):
                a = open(os.path.join(self.p, folder, file), 'r')
                self.corpus = self.corpus + (nltk.sent_tokenize(a.read()))
                self.word_freq_dict[file] = {}
                self.word_dict[file] = []
                self.word_freq_dict[file]['frequencies'], self.word_freq_dict[file]['total_words'], self.word_dict[file] = self.find_frequency()
                if str(file) == self.break_at:  # it takes a really fucking long time to run so I j did this
                    break

    def find_frequency(self):
        freq_dict = {}
        word_dict = []
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
                word_dict.append(token)

        # print("example of processed token:{}".format(tokens))  # prints example of word tokens can be taken out later
        return freq_dict, total_words, word_dict

aethism = TotalCorp(b='dont break sir')
aethism.create_corp()
