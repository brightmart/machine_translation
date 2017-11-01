# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import collections
import jieba
import codecs
import os
import word2vec
import codecs
from data_util import _PAD,_UNK,_GO,_EOS
#jieba.enable_parallel(10)
def _read_words(filename,chinese=True):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3: #python 3.0 or above
      return f.read().replace("\n", " %s " % _EOS).split()
    else: #python2.7
      #if not chinese:
      #    string_list=f.read().decode("utf-8").lower().replace("\n"," ").replace("."," . ").replace(","," ,").replace("?"," ? ").split()
      #else:
      read_content=f.read().decode("utf-8").replace("\n", " ")
      seg_sentence = jieba.cut(read_content)
      seg_sentence = " ".join(seg_sentence)
      string_list = seg_sentence.split()
      return string_list


def _build_vocab_en(word2vec_model_path, vocabulary_cn_path, vocabulary_size_cn):
    model = word2vec.load(word2vec_model_path, kind='bin')
    flag = os.path.exists(vocabulary_cn_path)
    print("vocab english. exist or not:",flag)
    if not flag:
        i=0
        with codecs.open(vocabulary_cn_path, 'a', 'utf8') as f:
            f.write(_PAD + "\n")  # 0
            f.write(_UNK + "\n")  # 1
            f.write(_GO + "\n")  # 2
            f.write(_EOS + "\n")  # 3
            i = i+4
            for vocab, _ in zip(model.vocab, model.vectors):
                if i<vocabulary_size_cn:
                    f.write(vocab + "\n")
                    i=i+1

def _build_vocab(filename, vocab_path, vocab_size):
  """Reads a file to build a vocabulary of `vocab_size` most common words.
   The vocabulary is sorted by occurrence count and has one word per line.
  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
    vocab_size: size of the vocablulary to generate.
  """
  flag=os.path.exists(vocab_path)
  print(filename,"file exists or not:",flag)
  if not flag: # if dict not exists vocab, create; otherwise,ignore
      data = _read_words(filename,chinese=True)
      counter = collections.Counter(data) #Create a new, empty Counter object.
      count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
      words, _ = list(zip(*count_pairs))
      words = words[:(vocab_size-4)] # (u'<EOS>', u'of', u'A', u'pair', u'on', u'-', u'Five', u'a'...)
      with codecs.open(vocab_path, 'a', 'utf8') as f: #open(vocab_path, "wb") as f:
          f.write(_PAD + "\n") #0
          f.write(_UNK+ "\n")  #1
          f.write(_GO + "\n")  #2
          f.write(_EOS + "\n") #3
          for word in words:
              f.write(word.strip()+"\n")

#filename='./data/train.zh'
#vocab_path='./data/vocabulary.zh'

#filename='./data/train.en'
#vocab_path='./data/vocabulary.en'
#_build_vocab(filename,vocab_path,vocab_size=100000)