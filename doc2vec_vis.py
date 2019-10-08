import gensim
import matplotlib
import matplotlib.pyplot as plt
import sys
import gensim.models.doc2vec as d2v
import wordcloud
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import axes3d, Axes3D




# load model
fn = sys.argv[1]
model = d2v.Doc2Vec.load(fn)

# gets words mose similar to a word, sorted by similarity
# takes string input
def get_most_similar(word):
    ranks = model.wv.most_similar(word, topn=100)
    return ranks


# returns a word cloud based on a word's similarities from model
def word_cloud_of_similarity(word):
    words = get_most_similar(word)
    dict = {}
    for pair in words:
        dict[pair[0]] = pair[1]

    return wordcloud.WordCloud(max_font_size=30, max_words=300, background_color="white").generate_from_frequencies(dict)


cloud = word_cloud_of_similarity("religion")
plt.figure()
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title('Words related to religion, based on doc2vec model', pad=20)
# plt.show()

vecs = model.docvecs
datapoints = [[]]
pca = PCA(n_components=3)
X = model[model.wv.vocab]
minMaxScaler = preprocessing.MinMaxScaler()
X_scaled = minMaxScaler.fit_transform(X)
result = pca.fit_transform(X_scaled)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Scaled projection of subset of document embedding space', pad=10, fontsize=10)
plt.scatter(result[:, 0], result[:, 1], result[:, 2])
plt.show()



