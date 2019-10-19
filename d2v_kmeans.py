from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d, Axes3D
import sys
import gensim.models.doc2vec as d2v
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


# load model
fn = sys.argv[1]
model = d2v.Doc2Vec.load(fn)

fn = sys.argv[2]
model2 = d2v.Doc2Vec.load(fn)

num_topics = 20
# ----------------------------- VOCAB 1 ---------------------------------
kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=100)
X = kmeans_model.fit(model.docvecs.doctag_syn0)
labels = kmeans_model.labels_.tolist()

colors = ['#28a509',
          '#f1089d',
          '#0b087a',
          '#a5e4e0',
          '#0cb4ab',
          '#410efe',
          '#b62627',
          '#8e72e1',
          '#ae7182',
          '#9c4cde']
color = [colors[i] for i in labels]
fig, ax = plt.subplots()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
datapoint = pca.transform(model.docvecs.doctag_syn0)
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color, marker='.')
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', c='black')
plt.title("KMeans clustering for Feature Vectors of Dov2Vec model (Vocab 1)")
plt.show()

# --------------------------- KMEANS 2 -------------------------------
kmeans_model = KMeans(n_clusters=5, init='k-means++', max_iter=100)
X = kmeans_model.fit(model2.docvecs.doctag_syn0)
labels = kmeans_model.labels_.tolist()


l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
datapoint = pca.transform(model.docvecs.doctag_syn0)
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color, marker='.')
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)

fig, ax = plt.subplots()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', c='black')
plt.title("KMeans clustering for Feature Vectors of Dov2Vec model (Vocab 2)")


plt.show()


