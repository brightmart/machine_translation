# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import collections
import jieba
import codecs
import os
from data_util import _PAD,_UNK,_GO,_EOS
#jieba.enable_parallel(40)
def _read_words(filename,chinese=True):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3: #python 3.0 or above
      return f.read().replace("\n", " %s " % EOS).split()
    else: #python2.7
      if not chinese:
          string_list=f.read().decode("utf-8").replace("\n","").split() # " %s " % EOS
      else:
          read_content=f.read().decode("utf-8").replace("\n", "") #" %s " % EOS
          seg_sentence = jieba.cut(read_content)
          seg_sentence = " ".join(seg_sentence)
          string_list = seg_sentence.split()
      return string_list


def _build_vocab(filename, vocab_path, vocab_size):
  """Reads a file to build a vocabulary of `vocab_size` most common words.
   The vocabulary is sorted by occurrence count and has one word per line.
  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
    vocab_size: size of the vocablulary to generate.
  """
  # if dict not exists vocab, create; otherwise,ignore
  flag=os.path.exists(vocab_path)
  print(filename,"file exists or not:",flag)
  if not flag:
      data = _read_words(filename,chinese=True)
      #print("data:",data) #('data:', [u'A', u'pair', u'of', u'red', u'-', u'crowned', u'cranes', u'have', u'staked', u'..
      counter = collections.Counter(data) #Create a new, empty Counter object.
      #print("counter:",counter) #counter will return a dict, like {'hello':5,'say':4}
      count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
      #print("count_pairs:",count_pairs) #counter will return a dict, like {'hello':5,'say':4}
      words, _ = list(zip(*count_pairs))
      #print("words:",words) # (u'<EOS>', u'of', u'A', u'pair', u'on', u'-', u'Five', u'a'...)
      words = words[:vocab_size] # (u'<EOS>', u'of', u'A', u'pair', u'on', u'-', u'Five', u'a'...)
      #print("words:",words) # (u'<EOS>', u'of', u'A', u'pair', u'on', u'-', u'Five', u'a'...)
      with codecs.open(vocab_path, 'a', 'utf8') as f: #open(vocab_path, "wb") as f:
          f.write(_PAD + "\n")
          f.write(_UNK+ "\n")
          f.write(_GO + "\n")
          f.write(_EOS + "\n")
          for word in words:
              f.write(word.strip()+"\n")
          #f.write("\n".join(words)) #write to file system. each word in a line.

#filename='./data/train.zh'
#vocab_path='./data/vocabulary.zh'

#filename='./data/train.en'
#vocab_path='./data/vocabulary.en'
#_build_vocab(filename,vocab_path,vocab_size=100000)