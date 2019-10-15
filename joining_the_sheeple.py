from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

CATEGORIES = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def fetch_data(categories=CATEGORIES):
    twenty_train = fetch_20newsgroups(subset='train', categories=categories)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories)

    return twenty_train, twenty_test


def create_vocabularies(twenty_train, twenty_test):
    count_vect_1 = CountVectorizer()  # vocabulary 1
    count_vect_2 = CountVectorizer(tokenizer=LemmaTokenizer(),
                                   strip_accents='unicode',
                                   lowercase=True,
                                   token_pattern=r'\b[a-zA-Z]{3,}\b')  # keeps words of 3 or more characters) # vocabulary 2

    X_train_counts_1 = count_vect_1.fit_transform(twenty_train.data)
    X_train_counts_2 = count_vect_2.fit_transform(twenty_train.data)

    X_test_counts_1 = count_vect_1.fit_transform(twenty_test.data)
    X_test_counts_2 = count_vect_2.fit_transform(twenty_test.data)

    return X_train_counts_1, X_train_counts_2, X_test_counts_1, X_test_counts_2


def create_vocabularies_tfidf(twenty_train, twenty_test):
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
    X = vectorizer.fit_transform(twenty_train.data)
    return X


def cluster(X, number_of_categories):
    kmeansc = KMeans(n_clusters=number_of_categories, init='k-means++', max_iter=100, n_init=1)
    kmeansc.fit(X)
    return kmeansc


train, test = fetch_data()

print("TFIDF: \n")
X_ = create_vocabularies_tfidf(train, test)
km = cluster(X_, len(CATEGORIES))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(train.target, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(train.target, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(train.target, km.labels_))

print("BOW 1\n")
X_, X_2, _, _ = create_vocabularies(train, test)
km = cluster(X_, len(CATEGORIES))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(train.target, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(train.target, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(train.target, km.labels_))
print("BOW 2\n")
km = cluster(X_2, len(CATEGORIES))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(train.target, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(train.target, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(train.target, km.labels_))


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
