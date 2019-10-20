import gensim
import matplotlib
import matplotlib.pyplot as plt
import sys
import gensim.models.doc2vec as d2v
import wordcloud
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import axes3d, Axes3D
import joblib

# load model
fn = sys.argv[1]
model = d2v.Doc2Vec.load(fn)

fn = sys.argv[2]
model2 = d2v.Doc2Vec.load(fn)

# gets words mose similar to a word, sorted by similarity
# takes string input
def get_most_similar(word):
    ranks = model.wv.most_similar(word, topn=50)
    return ranks


# returns a word cloud based on a word's similarities from model
def word_cloud_of_similarity(word):
    words = get_most_similar(word)
    dict = {}
    for pair in words:
        dict[pair[0]] = pair[1]

    return wordcloud.WordCloud(max_font_size=30, max_words=100, background_color="white").generate_from_frequencies(dict)

# ---------------------------------- Embedding Space ------------------------------------

cloud = word_cloud_of_similarity("religion")
plt.figure()
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title('Words related to religion, based on doc2vec model', pad=20)
# plt.show()

fig, ax = plt.subplots()
# model1 (good vocab)
vecs = model.docvecs
pca1 = PCA(n_components=3)
X = vecs.vectors_docs
minMaxScaler = preprocessing.MinMaxScaler()
X_scaled = minMaxScaler.fit_transform(X)
result = pca1.fit_transform(X_scaled)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.title('Scaled projection of document embedding space (Vocab 1)', pad=10, fontsize=10)
plt.scatter(result[:, 0], result[:, 1], result[:, 2], marker='o')
plt.show()

# model2 (bad vocab)
vecs = model2.docvecs
pca1 = PCA(n_components=3)
X = vecs.vectors_docs
minMaxScaler = preprocessing.MinMaxScaler()
X_scaled = minMaxScaler.fit_transform(X)
result = pca1.fit_transform(X_scaled)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.title('Scaled projection of document embedding space (Vocab 2)', pad=10, fontsize=10)
plt.scatter(result[:, 0], result[:, 1], result[:, 2], marker='o')
plt.show()


