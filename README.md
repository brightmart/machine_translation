# Machine Translation
Baseline model of machine translation using deep learning with lstm,cnn,attention,beam search and so on.

Usage
----------------------------------------------------------------------------------------------
#1. create a folder named 'data', and put sample data into this folder.

#2. train:   python a1_seq2seq_attention_train.py 

#3. predict: python a1_seq2seq_attention_predict.py

---------------------------------------
(below is optional)

#1. using pretrained word embedding:

if you want to use pretrained word embedding, you can run following command to generate word embedding file using word2vec:44ee
python a1_word2vec_train.py

#2. test model with toy data: learn to output reverse sequence of itself.

enable last line of a1_seq2seq_attention_model.py(that is to invoke test function) and run following command:

python a1_seq2seq_attention_model.py

What Dataset:
---------------------------------------------------------------------------------------------
In my experiment, i use dataset from <a href='http://challenger.ai/'>AI Challenger</a>, it has about 10 million pairs of training data, 8000 validation data, and 8000 testing data. translation direction is from english to chinese.


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

3.Attention is all you need

#Reference
------------------------------------------------------------------------------------------------
1.Neural Machine Translation by Jointly Learning to Align and Translate

2.Effective Approaches to Attention-based Neural Machine Translation

3.<a href='https://github.com/tensorflow/nmt'>Neural Machine Translation (seq2seq) Tutorial</a>

4.<a href='https://github.com/rsennrich/subword-nmt'>Subword Neural Machine Translation'</a>

for any question or suggestion, you can contact brightmart@hotmail.com
