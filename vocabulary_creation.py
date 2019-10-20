from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import time
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Religion', 'Comp', 'Rec', 'Science', 'Politics']
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    fig.tight_layout()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries

           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
    return ax


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'I', 'a', 'A', 'if'])
CATEGORIES = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']


def fetch_data(categories=CATEGORIES):
    """fetchs the 20 news group training and test data
    Args:
        categories (list): list of categories to fetch
    """
    twenty_train = fetch_20newsgroups(subset='train', categories=categories)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories)

    return twenty_train, twenty_test


WNL = WordNetLemmatizer()


def l(text):
    """lemmatize text. Does not work rn creates a list instead of the correct data
    args:
    text (string): the text from which to lemmatize"""
    return [WNL.lemmatize(t) for t in word_tokenize(text)]


class S(object):
    def __init__(self, wc):
        print("lets get this stats boi")
        self.wordcount = wc
        self.wordcount_processed = []
        self.sentence_count = 0


def remove_stopwords(text, stats):
    total_words = 0
    sentence = " "
    proc = 0
    for word in text.split():
        if word.lower() not in stop_words and word.isalnum():
            sentence += "{} ".format(word.lower())
            proc = proc + 1
        total_words += 1
    stats.wordcount.append(total_words)
    stats.wordcount_processed.append(proc)
    return sentence


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


def create_vocab1(data):
    """creates vocabulary one with stripped header footer and quoting and hopefully lemmatizing in the future
    Args (obj): 20newsgroup data
    """
    stats = S(wc=[])
    data.data = [strip_newsgroup_header(text) for text in data.data]
    data.data = [strip_newsgroup_footer(text) for text in data.data]
    data.data = [strip_newsgroup_quoting(text) for text in data.data]
    data.data = [remove_stopwords(text, stats) for text in data.data]
    plt.close()
    plt.hist(stats.wordcount, bins=20)
    print("number of Categories {}".format(len(data.data)))
    print("STD {}".format(np.std(np.array(stats.wordcount))))
    print("Mean {}".format(np.mean(np.array(stats.wordcount))))
    print("STD pre{}".format(np.std(np.array(stats.wordcount_processed))))
    print("Mean pre{}".format(np.mean(np.array(stats.wordcount_processed))))
    plt.title('Histogram of Wordcount per Document')
    plt.savefig('histogram.png')
    plt.close()
    plt.plot(range(len(data.data)), stats.wordcount)

    plt.title('Word Count vs. Document Number')
    plt.xlabel('Document Number')
    plt.ylabel('Word Count')
    plt.savefig('wordcount.png')
    plt.close()
    plt.plot(range(len(data.data)), stats.wordcount)
    plt.plot(range(len(data.data)), stats.wordcount_processed)
    plt.title('Word Count vs. Document Number Before and After Pre-Processing')
    plt.legend(['Original', 'Pre Processed'])
    plt.xlabel('Document Number')
    plt.ylabel('Word Count')
    plt.savefig('wordcount2.png')
    return data


def create_vocabularies_BOW(data, test=False):
    """Bag of Words model
    Args:
        data (obj): 20newsgroup object
    """
    count_vect_1 = CountVectorizer()  # vocabulary 1
    Y = []
    if test:
        X = count_vect_1.fit_transform(data.data)
        Y = count_vect_1.transform(test.data)
    else:
        X = count_vect_1.fit_transform(data.data)
    return X, Y


def create_vocabularies_tfidf(data, test=None):
    """TFIDF model
    Args:
        data (obj): 20newsgroup object
    """
    vectorizer = TfidfVectorizer()
    Y = []
    if test:
        X = vectorizer.fit_transform(data.data)
        Y = vectorizer.transform(test.data)
    else:
        X = vectorizer.fit_transform(data.data)
    return X, Y


def cluster(X, number_of_categories, data, name):
    """Performs k means clustering and prints homogeneity completeness and v measure
    Args:
        X (data): modeled data
        number_of_categories (int): number of clusters or categories to create
        data (obj): 20newsgroup object
        name (string): name to print out
    """
    km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    print(name)
    print("NMI: %0.3f" % normalized_mutual_info_score(data.target, km.labels_))


def train_(X_train, X_test, y_train, y_test):
    start = time.time()
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    end = time.time()
    predicted = classifier.predict(X_test)
    confusion_matrix(predicted, y_test)
    plot_confusion_matrix(y_test, predicted)
    print("Accuracy: " + str(classifier.score(X_test, y_test)) + ", Time duration: " + str(end - start))
    return classifier


def create_new_labels(labels):
    new_labels = []
    for c in labels:
        if c == 0 or c == 19 or c == 15:
            """religion"""
            new_labels.append(1)
        elif c <= 5 and c >= 1:
            new_labels.append(2)
        elif c >= 7 and c <= 10:
            new_labels.append(4)
        elif c >= 11 and c <= 14:
            new_labels.append(5)
        else:
            new_labels.append(3)
    return new_labels


# EXAMPLE:
def make_example(train, test):
    vocab2, vocab2_test = fetch_data()
    vocab1 = create_vocab1(train)
    vocab1_test = create_vocab1(test)
    vocab2.target = create_new_labels(vocab2.target)
    vocab1.target = create_new_labels(vocab1.target)
    vocab2_test.target = create_new_labels(vocab2_test.target)
    vocab1_test.target = create_new_labels(vocab1_test.target)
    cluster_ = True
    if cluster_:
        # bow = create_vocabularies_BOW(vocab1)
        bow, bow_test = create_vocabularies_BOW(vocab1, test=vocab1_test)

        print('Multinomial Naive Bayes: BOW Vocab1')
        train_(bow, bow_test, vocab1.target, vocab1_test.target)
        cluster(bow, len(CATEGORIES), train, "BOW VOCAB 1")
        # bow2 = create_vocabularies_BOW(vocab2)
        bow2, bow2_test = create_vocabularies_BOW(vocab2, test=vocab2_test)
        print('Multinomial Naive Bayes: BOW Vocab2')
        train_(bow2, bow2_test, vocab2.target, vocab2_test.target)
        cluster(bow2, len(CATEGORIES), train, "BOW VOCAB 2")

        tf_idf, tfidf_test = create_vocabularies_tfidf(vocab1, test=vocab1_test)
        print('Multinomial Naive Bayes: TFIDF Vocab1')
        train_(tf_idf, tfidf_test, vocab1.target, vocab1_test.target)
        cluster(tf_idf, len(CATEGORIES), train, "TFIDF VOCAB 1")

        tf_idf2, tfidf2_test = create_vocabularies_tfidf(vocab2, test=vocab2_test)
        print('Multinomial Naive Bayes: TFIDF Vocab2')
        train_(tf_idf2, tfidf2_test, vocab2.target, vocab2_test.target)
        cluster(tf_idf2, len(CATEGORIES), train, "TFIDF VOCAB 2")



def wordcloud(vocab1, vocab2):
    wordcloud = WordCloud().generate_from_text(' '.join(vocab1.data[0:1000]))
    plt.imshow(wordcloud)
    wordcloud2 = WordCloud().generate_from_text(' '.join(vocab2.data[0:1000]))
    plt.show()
    plt.imshow(wordcloud2)
    plt.show()


if (__name__ == '__main__'):
    train, test = fetch_data()
    make_example(train, test)

train, test = fetch_data()
vocab2, vocab2_test = fetch_data()
vocab1 = create_vocab1(train)
vocab1_test = create_vocab1(test)
vocab2.target = create_new_labels(vocab2.target)
vocab1.target = create_new_labels(vocab1.target)
vocab2_test.target = create_new_labels(vocab2_test.target)
vocab1_test.target = create_new_labels(vocab1_test.target)
