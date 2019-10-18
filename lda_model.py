
# -----------------------------------------------------------------
#                        Import Packages
# -----------------------------------------------------------------

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging

from joining_the_sheeple import fetch_data, create_vocab1

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# -----------------------------------------------------------------
#                         Prepare Stopwords
# -----------------------------------------------------------------

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

"""
# -----------------------------------------------------------------
#                         Import Dataset
# -----------------------------------------------------------------

df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

# -----------------------------------------------------------------
#                  Remove Emails and New Line Chars
# -----------------------------------------------------------------

# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# -----------------------------------------------------------------
#                Tokenize words and Clean-up text
# -----------------------------------------------------------------

"""
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

"""
data_words = list(sent_to_words(data))

# -----------------------------------------------------------------
#                Build bigram and trigram models
# -----------------------------------------------------------------

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# -----------------------------------------------------------------
#                  Do Lemmatization
# -----------------------------------------------------------------

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# -----------------------------------------------------------------
#                  Call Functions
# -----------------------------------------------------------------

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
"""

# -----------------------------------------------------------------
#                  Create Dictionary and Corpus
# -----------------------------------------------------------------

def buildModel(data):
    """
    Builds the gensim lda_model with the given data
    :return:
    """


    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus
    texts = data

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # -----------------------------------------------------------------
    #                  Build LDA Model
    # -----------------------------------------------------------------

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=20,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    return lda_model, corpus, id2word

# -----------------------------------------------------------------
#                  Visualize LDA Topics
# -----------------------------------------------------------------

def printLDAStats(lda_model, corpus):
    """
    Prints the stats of an lda model.
    """

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.


def getLDACoherence(lda_model, data, id2word):
    """
    Builds an LDA coherence model from an lda and prints its value.
    """
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)



if (__name__ == "__main__"):
    data = list(sent_to_words(create_vocab1(fetch_data()[0]).data))
    lda_model, corpus, id2word = buildModel(data)
    printLDAStats(lda_model, corpus)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
