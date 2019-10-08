import pre_process as pp

# Gensim
import gensim


# Create the tagged document needed for Doc2Vec
def create_tagged_documents(dict_of_list_of_words):
    documents = []
    for fn, doc in dict_of_list_of_words.items():
        documents.append(gensim.models.doc2vec.TaggedDocument(doc, [int(fn)]))

    return documents


# word dict contains tokenized list of words, with key = filename of doc
train_data = create_tagged_documents(pp.aethism.word_dict)

# Init the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(train_data, vector_size=5, min_count=2, epochs=40)
model.save("C:\\Users\\Zacha\\PycharmProjects\\dv_midterm\\d2v.model")
model_stored = gensim.models.doc2vec.Doc2Vec.load("C:\\Users\\Zacha\\PycharmProjects\\dv_midterm\\d2v.model")







