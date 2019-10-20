from pre_process import *  # I know this is not PEP8 don't come for me pls
import numpy as np

PATH_TO_PREPROCESSED = ""
folder = "alt.aethism"


# json.loads(zlib.decompress(b64decode(_['base64(zip(o))'])))

def s(pre_processed):
    sentence_length = []
    number_of_sentences = 0
    total_number_of_words = 0
    unique_words = 0
    number_of_books = len(pre_processed.keys())
    for book in pre_processed.keys():
        number_of_documents = len(pre_processed[book].keys())
        for idx, document in enumerate(pre_processed[book].keys()):
            all_words = pre_processed[book][document]['comma_seperated_words']

            for sentence in pre_processed[book][document]['sentence_vectors']:
                sentence_length.append(len(sentence))
                number_of_sentences += 1
        total_number_of_words = len(all_words)
        unique_words = len(set(all_words))

    return number_of_books, number_of_documents, number_of_sentences, total_number_of_words, unique_words, sentence_length


"""
for pick in os.listdir("pickles"):
    print("reading pickle: {}".format(pick))
    proc = pickle.load(open(os.path.join("pickles", pick), "rb"))
    print("processing pickle")
"""
c = TotalCorp(lemma=False, folder_name='alt.atheism', b='')
c.create_corp()
c2 = TotalCorp(lemma=False, folder_name='comp.graphics', b='')
c2.create_corp()
proc = {"aethism": c.word_freq_dict, "graphics": c2.word_freq_dict}
book_num, doc_num, sentence_num, total_words, unique_w, sentence_length = s(proc)
average_sentence_length = np.average(sentence_length)
std_sentence_length = np.std(sentence_length)
max_sentence_length = np.max(sentence_length)
min_sentence_length = np.min(sentence_length)
a = open('stats.txt', 'w')
a.write("Number of Books: {} \nNumber of Documents: {} \nTotal Words: {}  \n".format(book_num, doc_num, total_words,
                                                                                     unique_w))
a.write("Average Sentence Length: {} \nStandard Deviation: {} \nMax: {} \nMin: {} \n".format(average_sentence_length,
                                                                                             std_sentence_length,
                                                                                             max_sentence_length,
                                                                                             min_sentence_length))
a.close()
# bag_aethism = BagOfWords(proc['51060']['sentence_vectors'], proc['51060']['frequencies'])
# bag_aethism.create_sentence_vectors()

# TFIDF_aethism = TFIDF(proc['51060']['sentence_vectors'], proc)
# TFIDF_aethism.compute_tfidf()
