# -*- coding: utf-8 -*-
import tensorflow as tf

# 【该方法测试的时候使用】返回一个方法。这个方法根据输入的值，得到对应的索引，再得到这个词的embedding.
def extract_argmax_and_embed(embedding, output_projection=None):
    """
    Get a loop_function that extracts the previous symbol and embeds it. Used by decoder.
    :param embedding: embedding tensor for symbol
    :param output_projection: None or a pair (W, B). If provided, each fed previous output will
    first be multiplied by W and added B.
    :return: A loop function
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, 1) #得到对应的INDEX
        emb_prev = tf.gather(embedding, prev_symbol) #得到这个INDEX对应的embedding
        return emb_prev
    return loop_function

# version1:attention mechansion based on current time stamp and encode states to get attention vector, then concat attention vector with decode input by using feed forward layer.
# 如果是训练，使用训练数据的输入；如果是test,将t时刻的输出作为t+1时刻的s输入
def rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function,encode_source_states,hidden_size,scope=None):#3D Tensor [batch_size x attn_length x attn_size]
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].it is decoder input.
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].it is the encoded vector of input sentences, which represent 'thought vector'
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        encode_source_states: 3D Tensor [batch_size x attn_length x attn_size].it is represent input X.
        hidden_size: a scalar.
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
            (Note that in some cases, like basic RNN cell or GRU cell, outputs and
            states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        batch_size,sequence_length,_=encode_source_states.get_shape().as_list()
        encode_source_states_=tf.layers.dense(encode_source_states,hidden_size,use_bias=False) #[batch_size, sequence_length,hidden_size]. transform encode source states in advance, only once.
        outputs = []
        prev = None
        current_target_hidden_state = initial_state #shape:[batch_size x state_size]
        for i, inp in enumerate(decoder_inputs):#循环解码部分的输入。如sentence_length个[batch_size x input_size].如果是训练，使用训练数据的输入；如果是test, 将t时刻的输出作为t + 1 时刻的s输入
            if i==0:
                print(i, "inp:", inp)
            if loop_function is not None and prev is not None:#测试的时候：如果loop_function不为空且前一个词的值不为空，那么使用前一个的值作为RNN的输入
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
                    print(i, "inp:", inp)

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            #output,current_target_hidden_state=cell(inp,attention_vector)
            print("###inp:",inp,";current_target_hidden_state:",current_target_hidden_state)
            output, current_target_hidden_state = cell(inp, current_target_hidden_state)

            #1. the current target hidden state is compared with all source states to derive attention weights
            attention_weights=score_function(current_target_hidden_state,encode_source_states_,hidden_size) #[batch_size x sequence_length]
            #2. based on the attention weights we compute a context vector as the weighted average of the source states.
            context_vector=weighted_sum(attention_weights,encode_source_states) #[batch_size x attn_size]
            #3. combine the context vector with the current target hidden state to yield the final attention vector
            attention_vector=get_attention_vector(context_vector,current_target_hidden_state,hidden_size)
            #4. the attention vector is fed as an input to the next time step (input feeding).
            outputs.append(attention_vector) # 将输出添加到结果列表中
            if loop_function is not None:
                prev = attention_vector
    return outputs, current_target_hidden_state

# RNN的解码部分。
# version2: get attention vector using attention mechanism, invoke rnn together with decode input as final outupt.
# 如果是训练，使用训练数据的输入；如果是test,将t时刻的输出作为t+1时刻的s输入
def rnn_decoder_with_attention_TRAIL(decoder_inputs, initial_state, cell, loop_function,encode_source_states,hidden_size,scope=None):#3D Tensor [batch_size x attn_length x attn_size]
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].it is decoder input.
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].it is the encoded vector of input sentences, which represent 'thought vector'
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        encode_source_states: 3D Tensor [batch_size x attn_length x attn_size].it is represent input X.
        hidden_size: a scalar.
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
            shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
            (Note that in some cases, like basic RNN cell or GRU cell, outputs and
            states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        batch_size,sequence_length,_=encode_source_states.get_shape().as_list()
        encode_source_states_=tf.layers.dense(encode_source_states,hidden_size,use_bias=False) #[batch_size, sequence_length,hidden_size]. transform encode source states in advance, only once.
        outputs = []
        prev = None
        current_target_hidden_state = initial_state #shape:[batch_size x state_size]
        for i, inp in enumerate(decoder_inputs):#循环解码部分的输入。如sentence_length个[batch_size x input_size].如果是训练，使用训练数据的输入；如果是test, 将t时刻的输出作为t + 1 时刻的s输入
            if loop_function is not None and prev is not None:#测试的时候：如果loop_function不为空且前一个词的值不为空，那么使用前一个的值作为RNN的输入
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            #1. the current target hidden state is compared with all source states to derive attention weights
            attention_weights=score_function(current_target_hidden_state,encode_source_states_,hidden_size) #[batch_size x sequence_length]
            #2. based on the attention weights we compute a context vector as the weighted average of the source states.
            context_vector=weighted_sum(attention_weights,encode_source_states) #[batch_size x attn_size]
            #3. combine the context vector with the current target hidden state to yield the final attention vector
            attention_vector=get_attention_vector(context_vector,current_target_hidden_state,hidden_size)
            #4. the attention vector is fed as an input to the next time step (input feeding).
            print("inp:",inp,";attention_vector:",attention_vector) #('inp:', shape=(?, 2000) dtype=float32>, ';attention_vector:', shape=(1, 1000) dtype=float32>)
            output,current_target_hidden_state=cell(inp,attention_vector)
            outputs.append(output) # 将输出添加到结果列表中
            if loop_function is not None:
                prev = output
    return outputs, current_target_hidden_state

def get_attention_vector(context_vector,current_target_hidden_state,hidden_size):
    """
    get attention vector by concat context vector and current target hidden state, then use feed foward layer.
    attention_vector=tanh(Wc[c;h])
    :param context_vector: [batch_size x attn_size]
    :param current_target_hidden_state: [batch_size x state_size]
    :return get_attention_vector:[batch_size,hidden_size]
    """
    with tf.variable_scope("attention_vector"):
        attention_vector=tf.layers.dense(tf.concat([context_vector,current_target_hidden_state],-1),hidden_size,activation=tf.nn.tanh,use_bias=False) #[batch_size,hidden_size]
    return attention_vector #[batch_size,hidden_size]

def score_function(current_target_hidden_state,encode_source_states,hidden_size):
    """
    the current target hidden state is compared with all source states to derive attention weights. score=V_a.tanh(W1*h_t + W2_H_s)
    :param current_target_hidden_state: [batch_size x hidden_size]
    :param encode_source_states: [batch_size x sequence_length x hidden_size]
    :return: attention_weights: [batch_size x sequence_length]
    """
    with tf.variable_scope("score_function"):
        _, sequence_length, _=encode_source_states.get_shape().as_list()
        v= tf.get_variable("v_a", shape=[hidden_size,1],initializer=tf.random_normal_initializer(stddev=0.1))
        g=tf.get_variable("attention_g",initializer=tf.sqrt(1.0/hidden_size))
        b = tf.get_variable("bias", shape=[hidden_size], initializer=tf.zeros_initializer)

        # get part1: transformed current target hidden state
        current_target_hidden_state=tf.expand_dims(current_target_hidden_state,axis=1) #[batch_size,1,hidden_size]
        part1=tf.layers.dense(current_target_hidden_state, hidden_size,use_bias=False) #[batch_size,1,hidden_size]
        # additive and activated
        attention_logits=tf.nn.tanh(part1+encode_source_states+b) #[batch_size, sequence_length,hidden_size]
        # transform
        attention_logits=tf.reshape(attention_logits,(-1,hidden_size)) #[batch_size*sequence_length,hidden_size]
        normed_v=g*v*tf.rsqrt(tf.reduce_sum(tf.square(v))) #"Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks."normed_v=g*v/||v||,
        attention_weights=tf.reshape(tf.matmul(attention_logits,normed_v),(-1,sequence_length)) #[batch_size,sequence_length]
        # normalized
        attention_weights_max=tf.reduce_max(attention_weights,axis=1,keep_dims=True) #[batch_size,1]
        attention_weights=tf.nn.softmax(attention_weights-attention_weights_max) #[batch_size,sequence_length]
    return attention_weights #[batch_size x sequence_length]

def weighted_sum(attention_weights,encode_source_states):
    """
    weighted sum
    :param attention_weights:[batch_size x sequence_length]
    :param encode_source_states:[batch_size x sequence_length x attn_size]
    :return: weighted_sum: [batch_size, attn_size]
    """
    attention_weights=tf.expand_dims(attention_weights,axis=-1)      #[batch_size x sequence_length x 1]
    weighted_sum=tf.multiply(attention_weights,encode_source_states) #[batch_size x sequence_length x attn_size]
    weighted_sum=tf.reduce_sum(weighted_sum,axis=-1)                 #[batch_size x attn_size]
    return weighted_sum                                              #[batch_size x attn_size]