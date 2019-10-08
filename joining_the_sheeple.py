from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

count_vect_1 = CountVectorizer()  # vocabulary 1
count_vect_2 = CountVectorizer(tokenizer=LemmaTokenizer())  # vocabulary 2

# Bag of Words
X_train_counts_1 = count_vect_1.fit_transform(twenty_train.data)
X_train_counts_2 = count_vect_2.fit_transform(twenty_train.data)

X_test_counts_1 = count_vect_1.fit_transform(twenty_test.data)
X_test_counts_2 = count_vect_2.fit_transform(twenty_test.data)

kmeans1 = KMeans(n_clusters=4).fit(X_train_counts_1, twenty_train.target)
kmeans2 = KMeans(n_clusters=4).fit(X_train_counts_2, twenty_train.target)
predict1 = kmeans1.predict(X_test_counts_1)
predict2 = kmeans2.predict(X_test_counts_2)

mat = confusion_matrix(twenty_test.target, predict1.labels_)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=twenty_test.target_names,
            yticklabels=twenty_test.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

mat = confusion_matrix(twenty_test.target, predict2.labels_)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=twenty_test.target_names,
            yticklabels=twenty_test.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
