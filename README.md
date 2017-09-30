# Machine Translation
Baseline model of machine translation using deep learning with lstm,cnn,attention,beam search and so on.

Usage
----------------------------------------------------------------------------------------------
#1. create a folder named 'data', and put sample data into this folder.
#2. train:   python a1_seq2seq_attention_train.py 
#3. predict: python a1_seq2seq_attention_predict.py

(optional)
#1. using pretrained word embedding:
if you want to use pretrained word embedding, you can run following command to generate word embedding file using word2vec:44ee
python a1_word2vec_train.py

#2. test model with toy data:learn to output reverse sequence of itself.
enable last line and run following command:
python a1_seq2seq_attention_model.py

Description:
--------------------------------------------------------------------------------------------------
Implementation seq2seq with attention derived from NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE

I.Structure:

1)embedding 2)bi-GRU too get rich representation from source sentences(forward & backward). 3)decoder with attention.

II.Input of data:

there are two kinds of three kinds of inputs:1)encoder inputs, which is a sentence; 2)decoder inputs, it is labels list with fixed length;3)target labels, it is also a list of labels.

for example, labels is:"L1 L2 L3 L4", then decoder inputs will be:[_GO,L1,L2,L2,L3,_PAD]; target label will be:[L1,L2,L3,L3,_END,_PAD]. length is fixed to 6, any exceed labels will be trancated, will pad if label is not enough to fill.

III.Attention Mechanism:

    transfer encoder input list and hidden state of decoder

    calculate similiarity of hidden state with each encoder input, to get possibility distribution for each encoder input.

    weighted sum of encoder input based on possibility distribution.

    go though RNN Cell using this weight sum together with decoder input to get new hidden state

IV.How Vanilla Encoder Decoder Works:

the source sentence will be encoded using RNN as fixed size vector ("thought vector"). then during decoder:

    when it is training, another RNN will be used to try to get a word by using this "thought vector" as init state, and take input from decoder input at each timestamp. decoder start from special token "_GO". after one step is performanced, new hidden state will be get and together with new input, we can continue this process until we reach to a special token "_END". we can calculate loss by compute cross entropy loss of logits and target label. logits is get through a projection layer for the hidden state(for output of decoder step(in GRU we can just use hidden states from decoder as output).

    when it is testing, there is no label. so we should feed the output we get from previous timestamp, and continue the process util we reached "_END" TOKEN.

V.Notices:

    here i use two kinds of vocabularies. one is from words,used by encoder; another is for labels,used by decoder

    for vocabulary of lables, i insert three special token:"_GO","_END","_PAD"; "_UNK" is not used, since all labels is pre-defined.

#TODO
-----------------------------------------------------------------------------------------------
1.Google Neural Machine Translation
2.Convolutional Sequence to Sequence Learning

#Preference
------------------------------------------------------------------------------------------------
1.Neural Machine Translation by Jointly Learning to Align and Translate
