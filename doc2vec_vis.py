import gensim
import sys
import gensim.models.doc2vec as d2v

fn = sys.argv[1]
model = d2v.Doc2Vec.load(fn)

print(model.wv.most_similar("atheism"))
