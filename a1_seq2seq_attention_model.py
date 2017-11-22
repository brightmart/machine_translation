# -*- coding: utf-8 -*-
# seq2seq_attention: 1.word embedding 2.encoder 3.decoder(optional with attention). for more detail, please check:Neural Machine Translation By Jointly Learning to Align And Translate
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
import os
from a1_seq2seq import rnn_decoder_with_attention,extract_argmax_and_embed
#from data_util import _GO_ID,_END_ID

class seq2seq_attention_model:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,hidden_size, sequence_length_batch,is_training,decoder_sent_length=30,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001,use_beam_search=False):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.70) #0.5
        self.initializer = initializer
        self.decoder_sent_length=decoder_sent_length
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda
        self.use_beam_search=use_beam_search
        self.sequence_length_batch=sequence_length_batch

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")                 #x
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length],name="decoder_input")  #y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y_label") #y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,decoder_sent_length,self.num_classes]

        self.predictions = tf.argmax(self.logits, axis=2, name="predictions") #[batch_size,decoder_sent_length]
        self.accuracy = tf.constant(0.5)  # fuke accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)
        if not is_training:
            return
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def inference(self):
        """main computation graph here: 1.Word embedding. 2.Encoder with GRU 3.Decoder using GRU(optional with attention)."""
        # 1.word embedding
        embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  # [None, self.sequence_length, self.embed_size]

        # 2.encode with bi-directional GRU
        fw_cell =tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True) #rnn_cell.LSTMCell
        bw_cell =tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_words, dtype=tf.float32,#sequence_length: size `[batch_size]`,containing the actual lengths for each of the sequences in the batch
                                                          sequence_length=self.sequence_length_batch, time_major=False, swap_memory=True)
        encode_outputs=tf.concat([bi_outputs[0],bi_outputs[1]],-1) #should be:[None, self.sequence_length,self.hidden_size*2]

        # 3. decode with attention
        # decoder_inputs: embeding and split
        decoder_inputs=tf.nn.embedding_lookup(self.Embedding_label,self.decoder_input) #[batch_size,self.decoder_sent_length,embed_size]
        decoder_inputs = tf.split(decoder_inputs, self.decoder_sent_length,axis=1)  # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        decoder_inputs = [tf.squeeze(x, axis=1) for x in decoder_inputs]  # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=False)
        output_projection = (self.W_projection, self.b_projection)
        loop_function=extract_argmax_and_embed(self.Embedding_label,output_projection) if not self.is_training else None
        initial_state=tf.concat([bi_state[0][0],bi_state[1][0]],-1) #shape:[batch_size,hidden_size*2].last hidden state as initial state.bi_state[0] is forward hidden states; bi_state[1] is backward hidden states
        outputs, final_state = rnn_decoder_with_attention(decoder_inputs, initial_state, cell,loop_function, encode_outputs,self.hidden_size,scope=None)  # A list.length:decoder_sent_length.each element is:[batch_size x output_size]
        decoder_output = tf.stack(outputs, axis=1)  # decoder_output:[batch_size,decoder_sent_length,hidden_size]
        decoder_output = tf.reshape(decoder_output, shape=(-1, self.hidden_size))  # decoder_output:[batch_size*decoder_sent_length,hidden_size]
        with tf.name_scope("dropout"): #dropout as regularization
            decoder_output = tf.nn.dropout(decoder_output,keep_prob=self.dropout_keep_prob)  # shape:[-1,hidden_size]
        # 4. get logits
        with tf.name_scope("output"):
            print("###decoder_output:",decoder_output) # <tf.Tensor 'dropout/dropout/mul:0' shape=(12, 1000)
            logits = tf.matmul(decoder_output, self.W_projection) + self.b_projection  # logits shape:[batch_size*decoder_sent_length,self.num_classes]==tf.matmul([batch_size*decoder_sent_length,hidden_size*2],[hidden_size*2,self.num_classes])
            logits=tf.reshape(logits,shape=(self.batch_size,self.decoder_sent_length,self.num_classes)) #logits shape:[batch_size,decoder_sent_length,self.num_classes]
        return logits

    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits);#losses:[batch_size,self.decoder_sent_length]
            loss_batch=tf.reduce_sum(losses,axis=1)/self.decoder_sent_length #loss_batch:[batch_size]
            loss=tf.reduce_mean(loss_batch)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
            return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size*2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size*2])
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size*2],dtype=tf.float32) #,initializer=self.initializer
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size, self.num_classes],initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])


# test started: learn to output reverse sequence of itself.
def train():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 9+2 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 300
    embed_size = 1000 #100
    hidden_size = 1000
    is_training = True
    dropout_keep_prob = 1  # 0.5 #num_sentences
    decoder_sent_length=6
    l2_lambda=0.0001
    sequence_length_batch=[sequence_length]*batch_size
    model = seq2seq_attention_model(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size,hidden_size, sequence_length_batch,is_training,decoder_sent_length=decoder_sent_length,l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500):
            label_list=get_unique_labels()
            input_x = np.array([label_list],dtype=np.int32) #[2,3,4,5,6]
            label_list_original=copy.deepcopy(label_list)
            label_list.reverse()
            decoder_input=np.array([[0]+label_list],dtype=np.int32) #[[0,2,3,4,5,6]] #'0' is used to represent start token.
            input_y_label=np.array([label_list+[1]],dtype=np.int32) #[[2,3,4,5,6,1]] #'1' is used to represent end token.
            loss, acc, predict, W_projection_value, _ = sess.run([model.loss_val, model.accuracy, model.predictions, model.W_projection, model.train_op],
                                                     feed_dict={model.input_x:input_x,model.decoder_input:decoder_input, model.input_y_label: input_y_label,
                                                                model.dropout_keep_prob: dropout_keep_prob})
            if i%300==0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess,save_path,global_step=i)

            print(i,"loss:", loss, "acc:", acc, "label_list_original as input x:",label_list_original,";input_y_label:", input_y_label, "prediction:", predict)

def predict():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 9 + 2  # additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 300
    embed_size = 100
    hidden_size = 100
    is_training = False #THIS IS DIFFERENT FROM TRAIN()
    dropout_keep_prob = 1
    decoder_sent_length = 6
    l2_lambda = 0.0001
    model = seq2seq_attention_model(num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size, hidden_size, is_training,
                                    decoder_sent_length=decoder_sent_length, l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
        for i in range(100):
            label_list = get_unique_labels()
            input_x = np.array([label_list], dtype=np.int32)
            label_list_original = copy.deepcopy(label_list)
            label_list.reverse()
            decoder_input = np.array([[0] + [0]*len(label_list)],dtype=np.int32)
            acc, prediction = sess.run([ model.accuracy, model.predictions],
                             feed_dict={model.input_x: input_x, model.decoder_input: decoder_input, model.dropout_keep_prob: dropout_keep_prob})
            input_y_label = np.array([label_list + [1]],dtype=np.int32)
            print(i, "acc:", acc, "label_list_original as input x:", label_list_original,";input_y_label:", input_y_label, "prediction:", prediction)

def get_unique_labels():
    x=[2,3,4,5,6]
    random.shuffle(x)
    return x

#train()
#predict()