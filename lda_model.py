"""
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
"""


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
#%matplotlib inline

# Enable logging for gensim - optional
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


#Getting parsed data
from joining_the_sheeple import fetch_data, create_vocab1

#Packages for jupiter visualization
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Packages for words per doc visualization
import seaborn as sns

# Packages for word cloud visualization
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#Packages for words importance visualization
from collections import Counter

#Packages for sentence visualization
from matplotlib.patches import Rectangle

#Pretty colors
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])



#-----------------------------------------------------------------------
#                  Get data for LDA
#-----------------------------------------------------------------------

def getDataHayle():
    sentences = list(sent_to_words(create_vocab1(fetch_data()[0]).data))
    return [l for l in sentences if l]


def getDataMe():
    """
    Get and parse newsgroup data according to tutorial
    """

    # Read the 20 newsgroup data from a github
    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')


    #  Remove Emails and New Line Chars

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    data_words = list(sent_to_words(data))


    # Build bigram and trigram models

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


    # Do Lemmatization


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


    #  Call Functions

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized


def sent_to_words(sentences):
    """
    Breaks things up into sentences
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# -----------------------------------------------------------------
#                  Build the actual model
# -----------------------------------------------------------------

def buildModel(data):
    """
    Builds the gensim lda_model with the given data
    :return: the lda model, a corpus and an id-to-word dictionary
    """

    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=12,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    return lda_model, corpus, id2word


#-----------------------------------------------------------------------
#                     Get LDA Stats
#-----------------------------------------------------------------------


def format_topics_sentences(ldamodel, corpus, data):
    """
    Gets a dataframe for the lda model that has the dominant topic
    for each document and some valuable keywords.
    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(data)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Reset the topic index and add some column headers
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return df_dominant_topic


def printLDAStats(lda_model, corpus):
    """
    Prints the stats of an lda model.
    """

    # Print the Keywords for each topic
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



#-----------------------------------------------------------------------
#                     Make lda visualizations
#-----------------------------------------------------------------------


def makeJupiterVis(lda_model, corpus, id2word):
    """
    Builds an interactive html file for visualizing an lda model.
    """
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=id2word)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')


def makeWordsPerDocVisualization(df_dominant_topic, plotBreakup):
    """
    Makes a visualization for how many times dominant words show up in the various document in
    an LDA model.
    Pass this function the return of format_topics_sentences()
    :param plotBreakup is a tuple of rows and columns for each topic.
    """

    for j in range(0, 2):
        fig, axes = plt.subplots(2, 3, figsize=(10, 8), dpi=160, sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            idx = i + j * 6
            df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == idx, :]
            doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
            ax.set(xlim=(0, 1000), xlabel='Document Word Count')
            ax.set_ylabel('Number of Documents', color=cols[idx % len(cols)], fontsize=8)
            ax.tick_params(axis='y', labelcolor=cols[idx % len(cols)], color=cols[idx % len(cols)])
            ax.set_title('Topic: ' + str(idx), fontdict=dict(size=12, color=cols[idx % len(cols)]))
            ax.hist(doc_lens, bins=1000, color=cols[idx % len(cols)])
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                label.set_visible(False)
            if len(doc_lens) > 1:
                sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())

        fig.subplots_adjust(top=0.90)
        plt.xticks(np.linspace(0, 1000, 9))
        fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=16)
        fig.tight_layout(pad=4, w_pad=4, h_pad=6)
        plt.show()


def makeWordCloudVisualization(lda_model, words_in_cloud, plotBreakup):
    cloud = WordCloud(
        # stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=words_in_cloud,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False, num_topics=12, num_words=words_in_cloud)

    for j in range(0, 3):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

        for i, ax in enumerate(axes.flatten()):
            idx = i + j * 4
            fig.add_subplot(ax)
            topic_words = dict(topics[idx][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(idx), fontdict=dict(size=12))
            plt.gca().axis('off')

        plt.subplots_adjust(wspace=10, hspace=10)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()


def makeWordWeightImportanceVisualization(lda_model, data, plotBreakup):
    """
    Makes a visualization of the frequency and importance of each words in
    the lda topics.
    """
    topics = lda_model.show_topics(formatted=False, num_topics=12, num_words=20)
    data_flat = [w for w_list in data for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    for j in range(0, 3):
        # Plot Word Count and Weights of Topic Keywords
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, dpi=160)
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        for i, ax in enumerate(axes.flatten()):
            idx = i + j * 4
            ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == idx, :], color=cols[idx % len(cols)], width=0.5, alpha=0.3,
                   label='Word Count')
            ax_twin = ax.twinx()
            ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == idx, :], color=cols[idx % len(cols)], width=0.2,
                        label='Weights')
            ax.set_ylabel('Word Count', color=cols[idx % len(cols)])
            # ax_twin.set_ylim(0, 0.030)
            # ax.set_ylim(0, 3500)
            ax.set_title('Topic: ' + str(idx), color=cols[idx % len(cols)], fontsize=12)
            ax.tick_params(axis='y', left=False)
            ax.set_xticklabels(df.loc[df.topic_id == idx, 'word'], rotation=30, horizontalalignment='right', fontsize=7)
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')

        fig.tight_layout(pad=4, w_pad=2, h_pad=8)
        fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=16, y=1.05)
        plt.show()


def sentences_chart(lda_model, corpus, start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1]
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = []
            for wd, topic in wordid_topics:
                if len(topic) > 0:
                    word_dominanttopic.append((lda_model.id2word[wd], topic[0]))

            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=10, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectangle
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.2
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=10, color=mycolors[topics % len(mycolors)],
                            transform=ax.transAxes, fontweight=700)
                    word_pos = round(word_pos + .025 * len(word), 3) # to move the word for the next iter

                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=10, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=16, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()



#-----------------------------------------------------------------------
#                     Run Everything
#-----------------------------------------------------------------------

if (__name__ == "__main__"):
    data = getDataHayle()
    lda_model, corpus, id2word = buildModel(data)

    makeJupiterVis(lda_model, corpus, id2word)

    # df_dominant_topic = format_topics_sentences(lda_model, corpus, data)
    # makeWordsPerDocVisualization(df_dominant_topic, (2,5))
    #
    # makeWordCloudVisualization(lda_model, 20, (2,2))
    #
    # makeWordWeightImportanceVisualization(lda_model, data, (2,2))
    #
    # sentences_chart(lda_model, corpus)

