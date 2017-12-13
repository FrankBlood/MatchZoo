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

class MultiBiGRU(BasicModel):
    def __init__(self, config):
        super(MultiBiGRU, self).__init__(config)
        self.__name = 'MultiBiGRU'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MultiBiGRU] parameter check wrong')
        print('[MultiBiGRU] init done', end='\n')

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
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        rnn_with_hidden = Bidirectional(GRU(units=self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate'], return_sequences=True))
        rnn = Bidirectional(GRU(units=self.config['hidden_size']*2, dropout=self.config['dropout_rate'],
                                recurrent_dropout=self.config['dropout_rate'], return_sequences=False))
        q_rep = rnn_with_hidden(q_embed)
        q_rep = rnn(q_rep)
        show_layer_info('Bidirectional-GRU', q_rep)
        d_rep = rnn_with_hidden(d_embed)
        d_rep = rnn(d_rep)
        show_layer_info('Bidirectional-GRU', d_rep)

        merged = concatenate([q_rep, d_rep])
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

        model = Model(inputs=[query, doc], outputs=out_)
        model.summary()
        return model
