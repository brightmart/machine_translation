# -*- coding: utf-8 -*-
#prediction using model.process--->1.load data. 2.create session. 3.feed data. 4.predict
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import os
import codecs
from a1_seq2seq_attention_model import seq2seq_attention_model
from data_util import load_test_data,load_vocab,_GO,_PAD,_EOS,_UNK
from tflearn.data_utils import  pad_sequences

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","ckpt_ai_challenger_translation/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",30,"max sentence length")
tf.app.flags.DEFINE_integer("decoder_sent_length",30,"length of decoder inputs")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")
tf.app.flags.DEFINE_string("predict_target_file","ckpt_ai_challenger_translation/s2q_attention.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("data_en_test_path",'./data/test_a_20170923.sgm',"target file path for final prediction")
tf.app.flags.DEFINE_string("vocabulary_cn_path","./data/vocabulary.zh","path of traning data.")
tf.app.flags.DEFINE_string("vocabulary_en_path","./data/vocabulary.en","path of traning data.")

def main(_):
    #1.load test data
    vocab_cn, vocab_en = load_vocab(FLAGS.vocabulary_cn_path, FLAGS.vocabulary_en_path)
    test=load_test_data(FLAGS.data_en_test_path, vocab_en, FLAGS.decoder_sent_length)
    test = pad_sequences(test, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length

    #2.create session,model,feed data to make a prediction
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = seq2seq_attention_model(len(vocab_cn), FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                        FLAGS.decay_rate, FLAGS.sequence_length, len(vocab_en), FLAGS.embed_size,
                                        FLAGS.hidden_size, FLAGS.is_training,decoder_sent_length=FLAGS.decoder_sent_length, l2_lambda=FLAGS.l2_lambda)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        #feed data, to get logits
        number_of_test_data = len(test)
        print("number_of_test_data:", number_of_test_data)
        index = 0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        decoder_input=np.array([vocab_cn[_GO]] + [vocab_cn[_PAD]] * (FLAGS.decoder_sent_length - 1))
        decoder_input = np.reshape(decoder_input, [-1, FLAGS.decoder_sent_length])

        vocab_cn_index2word = dict([val, key] for key, val in vocab_cn.items())

        for start, end in zip(range(0, number_of_test_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_test_data + 1, FLAGS.batch_size)):
            predictions, logits = sess.run([model.predictions, model.logits], # logits:[batch_size,decoder_sent_length,self.num_classes]
                                           feed_dict={model.input_x: test[start:end],
                                                      model.decoder_input: decoder_input,
                                                      model.dropout_keep_prob: 1})  # 'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            output_sentence = get_label_using_logits(logits[0], predictions, vocab_cn_index2word, vocab_cn)
            # 7. write question id and labels to file system.
            predict_target_file_f.write(output_sentence+"\n")

        predict_target_file_f.close()

def get_label_using_logits(logits,predictions, vocab_cn_index2word, vocab_cn):
    print("logits.shape:", logits.shape) # logits:(30, 5897)
    selected_token_ids=[int(np.argmax(logit,axis=0)) for logit in logits]
    eos_index=vocab_cn[_EOS]
    if eos_index in selected_token_ids:
        eos_index = selected_token_ids.index(eos_index)
        selected_token_ids=selected_token_ids[0:eos_index]
    output_sentence = "".join([vocab_cn_index2word[index] for index in selected_token_ids])  # ) #cn_vocab[output] TODO TODO
    return output_sentence
if __name__ == "__main__":
    tf.app.run()