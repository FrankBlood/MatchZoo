# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.optimizers import Adam
from model import BasicModel

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from Match import *
from utility import *

class AMANDA(BasicModel):
    def __init__(self, config):
        super(AMANDA, self).__init__(config)
        self.__name = 'AMANDA'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                            'embed', 'embed_size', 'train_embed',  'vocab_size',
                            'hidden_size', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[AMANDA] parameter check wrong')
        print('[AMANDA] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'],
                              weights=[self.config['embed']], trainable=self.embed_trainable)

        sequence_level_encoding = Bidirectional(CuDNNLSTM(units=self.config['hidden_size'], return_sequences=True))

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        q_seq_encoded = sequence_level_encoding(q_embed)
        show_layer_info('Sequence-level Encoding', q_seq_encoded)
        d_seq_encoded = sequence_level_encoding(d_embed)
        show_layer_info('Sequence-level Encoding', d_seq_encoded)

        Attention_matrix = dot([d_seq_encoded, q_seq_encoded], axes=-1)
        show_layer_info('Attention matrix', Attention_matrix)
        Row_Softmax_matrix = TimeDistributed(Activation('softmax'))(Attention_matrix)
        # show_layer_info('Sequence-level Encoding', Row_Softmax_matrix)
        Aggregated_G_matrix = dot([Row_Softmax_matrix, q_seq_encoded], axes=[2, 1])
        show_layer_info('Aggregated G matrix', Aggregated_G_matrix)

        Query_Depend_Doc_S_matrix = concatenate([d_seq_encoded, Aggregated_G_matrix])
        show_layer_info('Sequence-level Encoding', Query_Depend_Doc_S_matrix)

        Query_Depend_Doc_LSTM_V_matrix = Bidirectional(CuDNNLSTM(units=self.config['hidden_size'],
                                           # dropout=self.config['dropout_rate'],
                                           # recurrent_dropout=self.config['dropout_rate'],
                                           return_sequences=False))(Query_Depend_Doc_S_matrix)
        show_layer_info('Sequence-level Encoding', Query_Depend_Doc_LSTM_V_matrix)

        merged = Dense(self.config['hidden_size'], activation='relu')(Query_Depend_Doc_LSTM_V_matrix)
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
