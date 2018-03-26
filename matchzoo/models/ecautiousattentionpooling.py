# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding,Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from Match import *
from utility import *

class ECautiousAttentionPooling(BasicModel):
    def __init__(self, config):
        super(ECautiousAttentionPooling, self).__init__(config)
        self.__name = 'ECautiousAttentionPooling'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ECautiousAttentionPooling] parameter check wrong')
        print('[ECautiousAttentionPooling] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.config['hidden_size']], bidirection[:, 0, self.config['hidden_size']:]])

        # 第一种匹配方式
        def full_matching(x):
            context, last_state = x
            # context = BatchNormalization()(context)
            # last_state = BatchNormalization()(last_state)
            matching = multiply([context, last_state])
            matching = Conv1D(filters=self.config['hidden_size'], kernel_size=1, activation='tanh')(matching)
            return matching

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        conv1d = Conv1D(filters=self.config['hidden_size'], kernel_size=3, padding='same', activation='tanh')
        convert = Dense(self.config['embed_size'], activation='tanh')
        transfer = Conv1D(filters=self.config['hidden_size'], kernel_size=1, padding='same', activation='tanh')
        extract = Conv1D(filters=self.config['hidden_size'], kernel_size=1, padding='same', activation='relu')
        rnn_with_seq = Bidirectional(GRU(units=self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                         recurrent_dropout=self.config['dropout_rate'], return_sequences=True))

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        q_conv = conv1d(q_embed)
        q_conv = Dropout(self.config['dropout_rate'])(q_conv)
        show_layer_info("Conv1D Q", q_conv)
        q_conv = Permute([2, 1])(q_conv)
        q_global_pool = GlobalMaxPooling1D()(q_conv)
        q_global_pool = Reshape([self.config['text1_maxlen'], 1])(q_global_pool)
        show_layer_info("Global Max Pooling Q", q_global_pool)
        q_global_pool = concatenate([q_embed, q_global_pool])
        show_layer_info("Max Pooling Q", q_global_pool)

        d_conv = conv1d(d_embed)
        d_conv = Dropout(self.config['dropout_rate'])(d_conv)
        show_layer_info("Conv1D D", d_conv)
        d_conv = Permute([2, 1])(d_conv)
        d_global_pool = GlobalMaxPooling1D()(d_conv)
        d_global_pool = Reshape([self.config['text2_maxlen'], 1])(d_global_pool)
        show_layer_info("Global Max Pooling D", d_global_pool)
        d_global_pool = concatenate([d_embed, d_global_pool])
        show_layer_info("Max Pooling D", d_global_pool)

        q_rep = rnn_with_seq(q_global_pool)
        show_layer_info('Bidirectional-GRU of Q', q_rep)
        d_rep = rnn_with_seq(d_global_pool)
        show_layer_info('Bidirectional-GRU of D', d_rep)

        q_rep_last_state = Lambda(lambda x: get_last_state(x))(q_rep)
        show_layer_info('Last state-Q-representation', q_rep_last_state)
        d_rep_last_state = Lambda(lambda x: get_last_state(x))(d_rep)
        show_layer_info('Last state-D-representation', d_rep_last_state)

        attention = concatenate([q_rep_last_state, d_rep_last_state])
        attention = Dense(self.config['hidden_size']*2, activation='tanh')(attention)
        # attention = Dropout(self.config['dropout_rate'])(attention)
        show_layer_info('Attention', attention)

        attention_q = dot([q_rep, attention], axes=-1, normalize=True)
        attention_q = Activation('softmax')(attention_q)
        show_layer_info('Attention of Q', attention_q)

        attention_d = dot([d_rep, attention], axes=-1, normalize=True)
        attention_d = Activation('softmax')(attention_d)
        show_layer_info('Attention of D', attention_d)

        # 对有hidden state的输出用attention加权求和
        q_rep = Permute([2, 1])(q_rep)
        d_rep = Permute([2, 1])(d_rep)

        new_q_rep = multiply([q_rep, attention_q])
        new_d_rep = multiply([d_rep, attention_d])
        new_q_rep = Permute([2, 1])(new_q_rep)
        new_d_rep = Permute([2, 1])(new_d_rep)

        new_q_rep = Lambda(lambda x: K.sum(x, axis=1))(new_q_rep)
        show_layer_info('Final representation of Q', new_q_rep)
        new_d_rep = Lambda(lambda x: K.sum(x, axis=1))(new_d_rep)
        show_layer_info('Final representation of D', new_d_rep)

        merged = concatenate([new_q_rep, new_d_rep])
        # merged = Dropout(self.config['dropout_rate'])(merged)
        show_layer_info('Aggression of Two texts', merged)
        merged = Dense(self.config['hidden_size'] * 2, activation='relu')(merged)
        merged = Dropout(self.config['dropout_rate'])(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.config['hidden_size'], activation='relu')(merged)
        merged = Dropout(self.config['dropout_rate'])(merged)
        merged = BatchNormalization()(merged)
        show_layer_info('Final representation of two texts', merged)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(merged)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(merged)
        show_layer_info('Dense', out_)

        # out_ = Dot(axes= [1, 1], normalize=True)([new_q_rep, new_d_rep])

        model = Model(inputs=[query, doc], outputs=out_)
        model.summary()
        return model
