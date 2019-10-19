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
import nltk
from sklearn.model_selection import train_test_split

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'I', 'a', 'A', 'if'])
CATEGORIES = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


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


def remove_stopwords(text):
    sentence = " "
    for word in text.split():
        if word.lower() not in stop_words and word.isalnum():
            sentence += "{} ".format(word.lower())
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
    data.data = [strip_newsgroup_header(text) for text in data.data]
    data.data = [strip_newsgroup_footer(text) for text in data.data]
    data.data = [strip_newsgroup_quoting(text) for text in data.data]
    data.data = [remove_stopwords(text) for text in data.data]
    print(data.data[0])
    # data.data = [l(text) for text in data.data]
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
    km = KMeans(n_clusters=number_of_categories, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    print(name)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(data.target, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(data.target, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(data.target, km.labels_))


def train_(X_train, X_test, y_train, y_test):
    start = time.time()
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    end = time.time()

    print("Accuracy: " + str(classifier.score(X_test, y_test)) + ", Time duration: " + str(end - start))
    return classifier


# EXAMPLE:
train, test = fetch_data()
vocab2 = train
vocab2_test = test
vocab1 = create_vocab1(train)
vocab1_test = create_vocab1(test)

# bow = create_vocabularies_BOW(vocab1)
bow, bow_test = create_vocabularies_BOW(vocab1, test=vocab1_test)

train_(bow, bow_test, vocab1.target, vocab1_test.target)
cluster(bow, len(CATEGORIES), train, "BOW VOCAB 1")
# bow2 = create_vocabularies_BOW(vocab2)
bow2, bow2_test = create_vocabularies_BOW(vocab2, test=vocab2_test)
train_(bow2, bow2_test, vocab2.target, vocab2_test.target)
cluster(bow2, len(CATEGORIES), train, "BOW VOCAB 2")

tf_idf, tfidf_test = create_vocabularies_tfidf(vocab1, test=vocab1_test)
train_(tf_idf, tfidf_test, vocab1.target, vocab1_test.target)
cluster(tf_idf, len(CATEGORIES), train, "TFIDF VOCAB 1")

tf_idf2, tfidf2_test = create_vocabularies_tfidf(vocab2, test=vocab2_test)
train_(tf_idf2, tfidf2_test, vocab2.target, vocab2_test.target)
cluster(tf_idf2, len(CATEGORIES), train, "TFIDF VOCAB 2")

"""
IGNORE THIS WAS TRYING SOMETHING DIDN'T REALLY WORK 
kmeans1 = KMeans(n_clusters=4).fit_transform(X_train_counts_1, twenty_train.target)
kmeans2 = KMeans(n_clusters=4).fit_transform(X_train_counts_2, twenty_train.target)
clf = MultinomialNB(alpha=.01)
clf.fit(X_train_counts_1, twenty_train.target)
predicted = clf.predict(X_test_counts_1)
mat = confusion_matrix(twenty_train.test, predicted)
# print(mat)
sns.heatmap(mat.T)
"""

"""
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
# TFIDF
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts_1)
X_train_tf = tf_transformer.transform(X_train_counts_1)
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts_2)
X_train_tf2 = tf_transformer.transform(X_train_counts_2)
kmeans1_tf = KMeans(n_clusters=4).fit(X_train_tf, twenty_train.target)
kmeans2_tf = KMeans(n_clusters=4).fit(X_train_tf2, twenty_train.target)
mat = confusion_matrix(twenty_train.target, kmeans1_tf.labels_)
sns.heatmap(mat.T,
            xticklabels=twenty_test.target_names,
            yticklabels=twenty_test.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
"""