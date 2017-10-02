# -*- coding: utf-8 -*-
import word2vec
import codecs

#词向量
data_path='./data/train.en' 
target_file="./data/ai_challenger_translation.bin-128"
def train():
	word2vec.word2vec(data_path, target_file, size=128, binary=1, min_count=20, verbose=True)

if __name__ == "__main__":
	train()
