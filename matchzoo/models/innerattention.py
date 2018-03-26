# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
import tensorflow as tf
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
sys.path.append('../matchzoo/models/')
from Match import *
from utility import *

from tensorflow import nn

# ----------------------------- cal attention -------------------------------
# input_q, input_a (batch_size, rnn_size, seq_len)
def cal_attention(input_q, input_a, U):
    batch_size = int(input_q.get_shape()[0])
    U = tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])
    G = tf.matmul(tf.matmul(input_q, U, True), input_a)
    delta_q = tf.nn.softmax(tf.reduce_max(G, 1), 1)
    delta_a = tf.nn.softmax(tf.reduce_max(G, 2), 1)

    return delta_q, delta_a

def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return cos_sim_q_a


# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])# (step, length of input for one step)
    #  do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    output = tf.reshape(output, [-1, width])
    return output


def cal_loss_and_acc(ori_cand, ori_neg):
    # the target function
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.2)
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses)
    # cal accurancy
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc


def ortho_weight(ndim):
    W = tf.random_normal([ndim, ndim], stddev=0.1)
    s, u, v = tf.svd(W)
    return u


def uniform_weight(nin, nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = tf.random_uniform(shape=[nin, nout], minval=-scale, maxval=-scale)
    return W


def BIGRU(x, hidden_size):

    '''
    功能：添加bidirectional操作
    :param x: [batch, height, width]   / [batch, step, embedding_size]
    :param hidden_size: 隐藏层节点个数
    :return: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]
    '''

    # input transformation
    input_x = tf.transpose(x, [1, 0, 2])
    # input_x = tf.reshape(input_x, [-1, w])
    # input_x = tf.split(0, h, input_x)
    input_x = tf.unstack(input_x)

    # define the forward and backward lstm cells
    gru_fw_cell = nn.rnn_cell.GRUCell(hidden_size)
    gru_bw_cell = nn.rnn_cell.GRUCell(hidden_size)
    output, _, _ = nn.static_bidirectional_rnn(gru_fw_cell, gru_bw_cell, input_x, dtype=tf.float32)

    # output transformation to the original tensor type
    output = tf.stack(output)
    output = tf.transpose(output, [1, 0, 2])
    return output


def slice(x, n, dim):
    if len(x.get_shape()) == 3:
        return x[:, :, n * dim:(n + 1) * dim]


def GRU_ATT(input_x, rnn_size, batch_size, vscope, summary_state=None, is_att=True):
    # input(batch_size, steps, embedding_size)
    num_steps = int(input_x.get_shape()[1])
    embedding_size = int(input_x.get_shape()[2])
    output = []  # (steps, batch_size, rnn_size)
    with tf.variable_scope(vscope):
        # define parameter
        W = tf.get_variable("W", initializer=tf.concat([uniform_weight(embedding_size, rnn_size),
                                                        uniform_weight(embedding_size, rnn_size)], 1))
        U = tf.get_variable("U", initializer=tf.concat([ortho_weight(rnn_size), ortho_weight(rnn_size)], 1))
        b = tf.get_variable("b", initializer=tf.zeros([2 * rnn_size]))
        Wx = tf.get_variable("Wx", initializer=uniform_weight(embedding_size, rnn_size))
        Ux = tf.get_variable("Ux", initializer=ortho_weight(rnn_size))
        bx = tf.get_variable("bx", initializer=tf.zeros([rnn_size]))
        M = tf.get_variable("M", initializer=tf.concat([uniform_weight(2 * rnn_size, rnn_size),
                                                        uniform_weight(2 * rnn_size, rnn_size)], 1))  # attention weight
        h_ = tf.zeros([batch_size, rnn_size])
        one = tf.fill([batch_size, rnn_size], 1.)
        state_below = tf.transpose(
            tf.matmul(input_x, tf.tile(tf.reshape(W, [1, embedding_size, 2 * rnn_size]), [batch_size, 1, 1])) + b,
            perm=[1, 0, 2])
        state_belowx = tf.transpose(
            tf.matmul(input_x, tf.tile(tf.reshape(Wx, [1, embedding_size, rnn_size]), [batch_size, 1, 1])) + bx,
            perm=[1, 0, 2])  # (steps, batch_size, rnn_size)
        for time_step in range(num_steps):
            preact = tf.matmul(h_, U)
            preact = tf.add(preact, state_below[time_step])
            if is_att and summary_state is not None:
                preact = preact + tf.matmul(summary_state, M)  # add attention

            print(preact.get_shape())
            print("")

            r = tf.nn.sigmoid(slice(preact, 0, rnn_size))
            u = tf.nn.sigmoid(slice(preact, 1, rnn_size))

            preactx = tf.matmul(h_, Ux)
            preactx = tf.multiply(preactx, r)
            preactx = tf.add(preactx, state_belowx[time_step])
            h = tf.tanh(preactx)

            h_ = tf.add(tf.multiply(u, h_), tf.multiply(tf.subtract(one, u), h))
            output.append(h_)
    output = tf.transpose(output, perm=[1, 0, 2])
    return output  # (batch_size, steps, rnn_size)


def BIGRU_ATT(input_x, rnn_size, batch_size, is_att=False, summary_state=None):

    with tf.variable_scope("FW") as fw_scope:

        h_encoder = GRU_ATT(input_x, rnn_size, batch_size, fw_scope, summary_state, is_att)
    with tf.variable_scope("BW") as bw_scope:
        h_encoder_rev = GRU_ATT(tf.reverse(input_x, [False, True, False]), rnn_size, batch_size, bw_scope, summary_state, is_att)

    h_encoder_rev = tf.reverse(h_encoder_rev, [False, True, False])
    output = tf.concat(2, [h_encoder, h_encoder_rev])
    return output
    # (batch_size, steps, rnn_size * 2)


class InnerAttention(BasicModel):
    def __init__(self, config):
        super(InnerAttention, self).__init__(config)
        self.__name = 'InnerAttention'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[InnerAttention] parameter check wrong')
        print('[InnerAttention] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        q_embed = Dropout(self.config['dropout_rate'])(q_embed)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        d_embed = Dropout(self.config['dropout_rate'])(d_embed)
        show_layer_info('Embedding', d_embed)

        def run_bigru_att(x, summary_state=None):
            ori_q = BIGRU_ATT(x, self.config['hidden_size'], 100, summary_state=summary_state)
            ori_q_feat = tf.nn.tanh(max_pooling(ori_q))
            return ori_q_feat

        ori_q_feat = Lambda(lambda x: run_bigru_att(x))(q_embed)
        cand_q_feat = Lambda(lambda x: run_bigru_att(x, summary_state=ori_q_feat))(d_embed)
        ori_cand = feature2cos_sim(ori_q_feat, cand_q_feat)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(ori_cand)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(ori_cand)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        model.summary()
        return model
