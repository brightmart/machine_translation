# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import collections
import jieba
import codecs
import os
from data_util import _PAD,_UNK,_GO,_EOS,experiment_mode_number
jieba.enable_parallel(30)
def _read_words(filename,chinese=True,experiment_mode=False):
  """Reads words from a file."""
  print("read content from file. started")
  with codecs.open(filename, 'r', 'utf8') as f: #tf.gfile.GFile(filename, "r") as f:
      if not chinese:
          print("start read all lines.english")
          string_list=f.readlines()
          print("start read all lines completed.english")
          dictt={}
          if experiment_mode:
              string_list = string_list[0:experiment_mode_number]
          for line in string_list:
              sub_list=line.decode("utf-8").strip().split()
              for element in sub_list:
                  element=element.strip()
                  value=dictt.get(element,1)
                  if value==1:
                      dictt[element]=1
                  else:
                      dictt[element]=value+1
              #string_list.decode("utf-8").replace("\n","").split() # " %s " % EOS
          print("read,split content.completed.english")
      else:
          print("start read all lines.chinese")
          read_content=f.read().decode("utf-8").replace("\n", "") #" %s " % EOS
          if experiment_mode:
              read_content=read_content[0:experiment_mode_number*30]
          print("read all lines.completed.chinese.")
          seg_sentence = jieba.cut(read_content)
          seg_sentence = " ".join(seg_sentence)
          string_list = seg_sentence.split()
          print("read,split content.completed.chinese")

      print("read content from file. ended")
      return string_list


def _build_vocab(filename, vocab_path, vocab_size,experiment_mode=False):
  """Reads a file to build a vocabulary of `vocab_size` most common words.
   The vocabulary is sorted by occurrence count and has one word per line.
  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
    vocab_size: size of the vocablulary to generate.
  """
  # if dict not exists vocab, create; otherwise,ignore
  print("start to create vocab.file:", filename)
  flag=os.path.exists(vocab_path)
  print(filename,"file exists or not:",flag)
  if not flag:
      data = _read_words(filename,chinese=True,experiment_mode=experiment_mode)
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
      print("end to create vocab")

#filename='./data/train.zh'
#vocab_path='./data/vocabulary.zh'

#filename='./data/train.en'
#vocab_path='./data/vocabulary.en'
#_build_vocab(filename,vocab_path,vocab_size=100000)