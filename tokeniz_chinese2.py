# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import jieba
import re
import time

_WORD_SPLIT = re.compile("([.,!?\":;，。！)(])")

def tokenize_save(data_cn_path,data_cn_path_processed):
    data_cn_object = codecs.open(data_cn_path, 'r', 'utf8')
    data_cn_object_target = codecs.open(data_cn_path_processed, 'w', 'utf8')

    #lines_cn = data_cn_object.readlines()
    for line in data_cn_object:
        #print(line,type(line))
        line=line.strip()
        seg_sentence = jieba.cut(line)
        seg_sentence = " ".join(seg_sentence)
        data_cn_object_target.write(seg_sentence.strip()+"\n")
        #if i%5000==0:
        #    print(i);print(seg_sentence)

def tokenize_save2(data_cn_path,data_cn_path_processed):
    jieba.enable_parallel(30)
    content = open(data_cn_path, "rb").read()
    t1 = time.time()
    words = "/ ".join(jieba.cut(content))
    t2 = time.time()
    tm_cost = t2 - t1
    log_f = open("data_cn_path_processed", "wb")
    log_f.write(words.encode('utf-8'))
    print('speed %s bytes/second' % (len(content) / tm_cost))

def read_trim_write(source,target):
    source_object = codecs.open(source, 'r', 'utf8')
    lines_cn = source_object.readlines()
    target_object = codecs.open(target, 'w', 'utf8')
    for i,line in enumerate(lines_cn):
        target_object.write(line.strip()+"\n")
        #if i%10000==0:
        #    print(i);print(line)
    target_object.close()
    source_object.close()

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

# calculate percentage of unk in english and chinese training data respectively.==>1.8%——1.6%
def calculate_percentage_unk(data_file,voc_file):
    #1.load vocabulary
    vocab_object = codecs.open(voc_file, 'r', 'utf8')
    vocab_list=vocab_object.readlines()
    vocab_list=[element.strip() for element in vocab_list]
    vocab_dict={}
    for i,vocab in enumerate(vocab_list):
        vocab_dict[vocab]=i
    print("len(vocab_dict):",len(vocab_dict))
    #print(vocab_dict)
    #2.load training data
    data_object = codecs.open(data_file, 'r', 'utf8')
    data_lines=data_object.readlines()
    total_count=0
    exist_count=0
    for i,line in enumerate(data_lines):
        line=line.strip()
        sub_list=line.split()
        total_count+=len(sub_list)
        for element in sub_list:
            value=vocab_dict.get(element,None)
            if value!=None:
                if int(value)<=60000:#(id<=60,000:'percentage:', 0.973200074848977, 'unk pert:', 0.02679992515102303, 'total_count:', 109185193, ';exist_count:', 106259038)
                    exist_count+=1
        if i%10000==0:
            print(i,";percentage:",float(exist_count)/float(total_count),";exist_count:",exist_count,";total_count:",total_count)
    percentage=exist_count/total_count
    print("percentage:",float(exist_count)/float(total_count),"unk pert:",1-float(exist_count)/float(total_count),"total_count:",total_count,";exist_count:",exist_count)
    return percentage

def preprocess_english(source_file,target_file):
    source_file_object = codecs.open(source_file, 'r', 'utf8')
    source_lines=source_file_object.readlines()

    target_file_object = codecs.open(target_file, 'w', 'utf8')

    for i,line in enumerate(source_lines):
        line=line.lower().replace("\n","").replace("."," . ").replace(","," ,").replace("?"," ? ")
        target_file_object.write(line+"\n")

    source_file_object.close()
    target_file_object.close()

def stat_length(source_file ):
    source_file_object = codecs.open(source_file, 'r', 'utf8')
    source_lines=source_file_object.readlines()
    total_length=0
    total_lines=len(source_lines)
    for i,line in enumerate(source_lines):
        total_length+=len(line.split())
    average_length=float(total_length)/float(total_lines)
    print("average_length:",average_length,";total_length:",total_length)
    return average_length

def stat_length_distribution(source_file ):
    source_file_object = codecs.open(source_file, 'r', 'utf8')
    source_lines=source_file_object.readlines()
    total_length=0
    total_lines=len(source_lines)
    dict_length={5:0,10:0,15:0,20:0,25:0,30:0,35:0}
    for i,line in enumerate(source_lines):
        length=len(line.strip().split())
        if length<=5:
            dict_length[5]=dict_length[5]+1
        elif length<=10:
            dict_length[10] = dict_length[10] + 1
        elif length<=15:
            dict_length[15] = dict_length[15] + 1
        elif length<=20:
            dict_length[20] = dict_length[20] + 1
        elif length<=25:
            dict_length[25] = dict_length[25] + 1
        elif length<=30:
            dict_length[30] = dict_length[30] + 1
        else:
            dict_length[35] = dict_length[35] + 1

    print("dict_length:",dict_length)
    total_length=0
    for k,v in dict_length.items():
        total_length+=v
    dict_length2={x:float(dict_length[x])/float(total_length) for x in dict_length.keys()}
    print("dict_length2:",dict_length2)
    return dict_length

#data_cn_path='train.zh'
#data_cn_path_processed='train_target2.zh'
#tokenize_save2(data_cn_path,data_cn_path_processed)
#source='test.zh'
#target='test.zh.tokenized'
#tokenize_save(source,target)

#data_file='train.zh'
#voc_file='vocabulary.zh'
#calculate_percentage_unk(data_file,voc_file) #(9900000, ';percentage:', 0.9840632628483765, ';exist_count:', 107411440, ';total_count:', 109150950)

#data_file='train.en'
#voc_file='vocabulary.en'
#calculate_percentage_unk(data_file,voc_file) #1-0.982=0.0018=1.8%

data_file='dev.zh'
voc_file='vocabulary.zh'
calculate_percentage_unk(data_file,voc_file) #1-0.99=0.01=1.0%

#data_file='dev.en'
#voc_file='vocabulary.en'
#calculate_percentage_unk(data_file,voc_file) #1-0.716=28.3%

#data_file='dev.en.processed'
#voc_file='vocabulary.en'
#calculate_percentage_unk(data_file,voc_file) #1-0.994=0.6%



#data_file='test.en'
#voc_file='vocabulary.en'
#calculate_percentage_unk(data_file,voc_file) #1-0.709=29.1%

#data_file='test_a_20170923.txt'
#voc_file='vocabulary.en'
#calculate_percentage_unk(data_file,voc_file) #1-0.994=0.65%

#source_file='test.en'
#target_file='test.en.processed'
#preprocess_english(source_file,target_file)

#source_file='train.en'
#stat_length(source_file)

#source_file='train.zh'
#stat_length(source_file)

#source_file='train.en'
#stat_length_distribution(source_file)  #most less than 10 and 15
#{35: 0.034, 5: 0.09, 10: 0.46, 15: 0.27, 20: 0.085, 25: 0.034, 30: 0.022})

#source_file='train.zh' #most less than 10 and 15
#{35: 0.0317, 5: 0.12, 10: 0.50, 15: 0.22, 20: 0.066, 25: 0.03, 30: 0.02}
#stat_length_distribution(source_file)


