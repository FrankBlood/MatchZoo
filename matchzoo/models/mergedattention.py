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

class MergedAttention(BasicModel):
    def __init__(self, config):
        super(MergedAttention, self).__init__(config)
        self.__name = 'MergedAttention'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MergedAttention] parameter check wrong')
        print('[MergedAttention] init done', end='\n')

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

        def Represent(embedded_sequence):
            cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(embedded_sequence)
            cnn = GlobalMaxPooling1D()(cnn)
            cnn = Dense(30, activation='relu')(cnn)
            return cnn

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        rnn_with_seq = Bidirectional(GRU(units=self.config['hidden_size'], dropout=self.config['dropout_rate'], return_sequences=True))
        q_rep = rnn_with_seq(q_embed)
        show_layer_info('Bidirectional-GRU', q_rep)
        d_rep = rnn_with_seq(d_embed)
        show_layer_info('Bidirectional-GRU', d_rep)

        q_rep_last_state = Lambda(lambda x: get_last_state(x))(q_rep)
        show_layer_info('Last state-Q-representation', q_rep_last_state)
        d_rep_last_state = Lambda(lambda x: get_last_state(x))(d_rep)
        show_layer_info('Last state-D-representation', d_rep_last_state)

        attention = concatenate([q_rep_last_state, d_rep_last_state])
        attention = Dense(self.config['hidden_size']*2, activation='tanh')(attention)
        show_layer_info('Attention', attention)

        attention_q = dot([q_rep, attention], axes=-1)
        attention_q = Activation('softmax')(attention_q)
        show_layer_info('Attention of Q', attention_q)

        attention_d = dot([d_rep, attention], axes=-1)
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

        merged_rnn = concatenate([new_q_rep, new_d_rep])
        show_layer_info('Aggression of Two texts', merged_rnn)
        merged_rnn = Dense(self.config['hidden_size'] * 2, activation='relu')(merged_rnn)
        merged_rnn = Dropout(self.config['dropout_rate'])(merged_rnn)
        merged_rnn = BatchNormalization()(merged_rnn)

        cnn_q = Lambda(lambda x: Represent(x))(q_embed)
        cnn_d = Lambda(lambda x: Represent(x))(d_embed)

        merged_cnn = concatenate([cnn_q, cnn_d])
        show_layer_info('Aggression of Two texts', merged_cnn)
        merged_cnn = Dense(self.config['hidden_size'] * 2, activation='relu')(merged_cnn)
        merged_cnn = BatchNormalization()(merged_cnn)
        merged_cnn = Dropout(self.config['dropout_rate'])(merged_cnn)

        merged = concatenate([merged_rnn, merged_cnn])
        merged = Dense(self.config['hidden_size']*2, activation='relu')(merged)
        merged = Dropout(self.config['dropout_rate'])(merged)
        merged = BatchNormalization()(merged)
        show_layer_info('Final representation of two classes of Network', merged)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(merged)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(merged)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        model.summary()
        return model
