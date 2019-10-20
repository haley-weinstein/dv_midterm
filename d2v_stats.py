import gensim
from sklearn.metrics import normalized_mutual_info_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

import d2v_kmeans as km
from sklearn.decomposition import PCA
import gensim.models.doc2vec as d2v
import numpy
import sys
import joining_the_sheeple as vcb

# Create the tagged document needed for Doc2Vec
def create_tagged_documents(vocab):
    documents = []
    sentence = []
    i = 0
    for line in vocab:
        for word in line.split():
            sentence.append(word)

        documents.append(gensim.models.doc2vec.TaggedDocument(sentence, [i]))
        i += 1
        sentence = []

    return documents


def calculate_nmi(kmeans, d2v_model):
    pca1d = PCA(n_components=1).fit_transform(d2v_model.docvecs.doctag_syn0)
    np = numpy.array(pca1d)
    np1d = numpy.ravel(np)
    print("NMI: %0.3f" % normalized_mutual_info_score(np1d, kmeans.labels_))


def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors


# load models
fn = sys.argv[1]
model1 = d2v.Doc2Vec.load(fn)

fn = sys.argv[2]
model2 = d2v.Doc2Vec.load(fn)

vocab1 = vcb.vocab1["data"]
vocab2 = vcb.vocab2["data"]

# train and analyze
train_docs1 = create_tagged_documents(vocab1[:2000])
train_docs2 = create_tagged_documents(vocab2[:2000])

test_docs1 = create_tagged_documents(vocab1[2001:4000])
test_docs2 = create_tagged_documents(vocab2[2001:4000])

calculate_nmi(km.kmeans_model1, model1)
calculate_nmi(km.kmeans_model2, model2)


