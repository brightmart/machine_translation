# -*- coding: utf-8 -*-
import codecs
import jieba
import re
import os
import pickle
import copy
import random
from tflearn.data_utils import to_categorical, pad_sequences

_PAD="_PAD"
_UNK="UNK"
_GO="_GO"
_EOS="_EOS"
experiment_mode_number=1000*1000
jieba.enable_parallel(30)
def load_data(data_folder,data_cn_path,data_en_path,vocab_cn, vocab_en,data_cn_valid_path,data_en_valid_path,sequence_length,experiment_mode=False,valid_portion=0.05): #vocab_cn:{token:index}
    """
    load raw data and vocabularies of chinese and english,then process it as training dataset.
    :return: train,valid,test.train:(X,y).
             X:a list. each line is a list of indices.
             y:a list. each line is a list of indices.
    """
    print("load_data.started.")
    cache_path=data_folder+'/train_valid.pik'
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as data_f:
            print("training and validation dataset exists. going to use exists one.")
            train, valid=pickle.load(data_f)
            return train,valid
    #1.load data of chinese
    data_cn_object = codecs.open(data_cn_path, 'r', 'utf8')
    lines_cn=data_cn_object.readlines()
    if experiment_mode:
        lines_cn=lines_cn[0:experiment_mode_number]
    #2.load data of english
    data_en_object = codecs.open(data_en_path, 'r', 'utf8')
    lines_en=data_en_object.readlines()
    if experiment_mode:
        lines_en=lines_en[0:experiment_mode_number]
    print("load raw data.completed.")
    #3.tokenize english data, replace token with index, and add to X
    X=[] #english
    print("length of enlgish of training:", len(lines_en))
    for i, line_en in enumerate(lines_en):# line_en: 'A bunch of fellas sitting in the middle of this field.'
        x=line_to_index_list(line_en, vocab_en, sequence_length,jieba=False)
        x.reverse() #reverse order of X
        X.append(x)

    #4. tokenize chinese data, replace token with index, and add to Y
    Y_input=[] #chinese
    Y_output=[]
    print("length of chinese of training:", len(lines_cn))
    for i,line_cn in enumerate(lines_cn):# line_cn:我们在自愿的基础上组织了流动送餐服务。
        y = line_to_index_list(line_cn, vocab_cn,sequence_length, jieba=True)
        y_original=copy.deepcopy(y) #save original
        y.insert(0,vocab_cn[_GO])       #insert START token, it can be used as input of decoder.
        Y_input.append(y)

        y_original.append(vocab_cn[_EOS])#insert END token, so it can be used as output of decoder.
        Y_output.append(y_original)

    # shuffle training data
    c=list(zip(X,Y_input,Y_output))
    random.shuffle(c)
    X[:], Y_input[:], Y_output[:]=zip(*c)
    train=(X,Y_input,Y_output) #need save train data to cache file to avoid generate training data again and again which takes lots of time.

    #5. read validation dataset
    X_valid=[]
    line_list_valid_en=read_sgm_file_as_list(data_en_valid_path)
    print("length of english of validation:", len(line_list_valid_en))
    for i, line_en in enumerate(line_list_valid_en):# line_en: 'A bunch of fellas sitting in the middle of this field.'
        x_valid = line_to_index_list(line_en, vocab_en, sequence_length,jieba=False)
        X_valid.reverse() #reverse order of X_valid
        X_valid.append(x_valid)

    line_list_valid_cn = read_sgm_file_as_list(data_cn_valid_path)
    Y_valid_input=[]
    Y_valid_output=[]
    print("length of chinese of validation:", len(line_list_valid_cn))
    for i,line_cn in enumerate(line_list_valid_cn):# line_cn:我们在自愿的基础上组织了流动送餐服务。
        y_valid = line_to_index_list(line_cn, vocab_cn,sequence_length, jieba=True)
        y_valid_original=copy.deepcopy(y_valid)
        y_valid.insert(0,vocab_cn[_GO])
        Y_valid_input.append(y_valid)

        y_valid_original.append(vocab_cn[_EOS])
        Y_valid_output.append(y_valid_original)
    valid = (X_valid,Y_valid_input,Y_valid_output)

    #6. save to file system as cahce file
    with open(cache_path, 'a') as data_f:
        pickle.dump((train,valid), data_f)
    print("load data completed.")
    return train,valid

def load_test_data(data_en_test_path,vocab_en,sequence_length):
    X_test=[]
    line_list_test_en=read_sgm_file_as_list(data_en_test_path)
    print("length of english of test:", len(line_list_test_en))
    for i, line_en in enumerate(line_list_test_en):# line_en: 'A bunch of fellas sitting in the middle of this field.'
        x_test= line_to_index_list(line_en, vocab_en, sequence_length,jieba=False)
        X_test.reverse() #reverse order of X_valid
        X_test.append(x_test)
    return X_test

def load_vocab(vocabulary_cn_path,vocabulary_en_path,experiment_mode=False):
    #3. load vocabulary of chinese
    vocabulary_cn_object = codecs.open(vocabulary_cn_path, 'r', 'utf8')
    vocabulary_cn_lines=vocabulary_cn_object.readlines()
    vocab_cn={}
    for i,voc in enumerate(vocabulary_cn_lines):
        voc=voc.strip()
        vocab_cn[voc]=i

    #4. load vocabulary of english
    vocabulary_en_object = codecs.open(vocabulary_en_path, 'r', 'utf8')
    vocabulary_en_lines=vocabulary_en_object.readlines()
    vocab_en={}
    for i,voc in enumerate(vocabulary_en_lines):
        voc=voc.strip()
        vocab_en[voc]=i
    return vocab_cn,vocab_en

def line_to_index_list(line,voc_dict,sequence_length,jieba=False):
    #print("source:");#'一对丹顶鹤正监视着它们的筑巢领地'
    #print(line);
    token_list = basic_tokenizer(line,jieba_flag=jieba)
    unk_index = voc_dict[_UNK]
    pad_index = voc_dict[_PAD]
    #print("unk_index:",unk_index,";pad_index:",pad_index)
    x = [voc_dict.get(e, unk_index) for e in token_list]
    if not jieba: #if jieba=False, mean it is english chinese. that is from source side. no need to append START OR END TOKEN, so there is no need to left a position for these special token.
        sequence_length=sequence_length+1
    if len(x)>=sequence_length-1:
        x=x[0:sequence_length-1]
        #print(" after.line_to_index_list--->",x)
        return x
    else:
        x_new=[pad_index]*(sequence_length-1)
        for i,e in enumerate(x):
            #print(i,e)
            x_new[i]=e
        #print(" after.line_to_index_list--->")
        #print(x_new)
        return x_new

def read_sgm_file_as_list(file_path):
    """
    :param file_path:
    :return: a list. each element is a line.
    """
    file_object = codecs.open(file_path, 'r', 'utf8')
    lines=file_object.readlines()
    listt=[]
    for i,line in enumerate(lines):
        if "<seg id=" in line:
            #line:'<seg id="10"> 我尊重这点，并且会不惜一切保护隐私不被侵犯。 </seg>'
            line= line[line.find(">")+1:line.find("</seg>")].strip()
            #print(i,line)
            listt.append(line)
    return listt


_WORD_SPLIT = re.compile("([.,!?\":;，。！)(])")
def basic_tokenizer(sentence, jieba_flag=False):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  if jieba_flag:
    tokens = list([w.lower() for w in jieba.cut(sentence) if w not in [' ']])
    #seg_sentence = jieba.cut(sentence)
    #seg_sentence = " ".join(seg_sentence)
    #tokens = seg_sentence.split()
  else:
    words = []
    for space_separated_fragment in sentence.strip().split():
      words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    tokens = [w.lower() for w in words if w]
  return tokens

#file_path='./data/valid.en-zh.zh.sgm'
#read_sgm_file_as_list(file_path)


