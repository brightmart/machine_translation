from collections import Counter
from nltk.corpus import brown
import tensorflow as tf
import time
import pickle
from nltk import sent_tokenize, word_tokenize
string_corpus=''
print("start...")
cache_path = 'data/vocabulary.en.nltk'


def generate_voc_en():
    with tf.gfile.GFile('train.en', "r") as f:
        string_corpus = f.read().decode("utf-8").lower().replace("\n"," ")
    print("end...read")
    start = time.time();
    fdist = Counter(word_tokenize(string_corpus));
    with open(cache_path, 'a') as data_f:
        pickle.dump(fdist, data_f)
    end = time.time() - start

def load_voc_en():
    with open(cache_path, 'r') as data_f:
        voc_en_dict = pickle.load(data_f)
        print("type(voc_en_dict):",type(voc_en_dict),len(voc_en_dict))
        for i,e in enumerate(voc_en_dict):
            print(e)
            if i>1000:
                break

load_voc_en()

