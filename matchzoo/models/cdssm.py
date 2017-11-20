
# -*- coding=utf-8 -*-
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, Merge, Lambda, Permute
from keras.layers import Convolution1D, MaxPooling1D, Reshape, Dot
from keras.activations import softmax

from model import BasicModel
import sys
sys.path.append('../matchzoo/utils/')
from utility import *


class CDSSM(BasicModel):
    def __init__(self, config):
        super(CDSSM, self).__init__(config)
        self.__name = 'CDSSM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'vocab_size', 'embed_size',
                   'filters', 'kernel_size', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[CDSSM] parameter check wrong')
        print '[CDSSM] init done'

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('filters', 32)
        self.set_default('kernel_size', 3)
        self.set_default('hidden_sizes', [300, 128])
        self.set_default('dropout_rate', 0.)
        self.config.update(config)

    def build(self):
        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.config['hidden_sizes'])
            assert num_hidden_layers > 0
            if num_hidden_layers == 1:
                seq.add(Dense(self.config['hidden_sizes'][0], input_shape=(input_dim,)))
            else:
                seq.add(Dense(self.config['hidden_sizes'][0], activation='relu', input_shape=(input_dim,)))
                for i in range(num_hidden_layers - 2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='relu'))
                    seq.add(Dropout(self.config['dropout_rate']))
                seq.add(Dense(self.config['hidden_sizes'][num_hidden_layers-1]))
                seq.add(Dropout(self.config['dropout_rate']))
            return seq
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        wordhashing = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = wordhashing(query)
        show_layer_info('Embedding', q_embed)
        d_embed = wordhashing(doc)
        show_layer_info('Embedding', d_embed)
        conv1d = Convolution1D(self.config['filters'], self.config['kernel_size'], padding='same', activation='relu')
        q_conv = conv1d(q_embed)
        show_layer_info('Convolution1D', q_conv)
        d_conv = conv1d(d_embed)
        show_layer_info('Convolution1D', d_conv)
        q_pool = MaxPooling1D(self.config['text1_maxlen'])(q_conv)
        show_layer_info('MaxPooling1D', q_pool)
        q_pool_re = Reshape((-1,))(q_pool)
        show_layer_info('Reshape', q_pool_re)
        d_pool = MaxPooling1D(self.config['text2_maxlen'])(d_conv)
        show_layer_info('MaxPooling1D', d_pool)
        d_pool_re = Reshape((-1,))(d_pool)
        show_layer_info('Reshape', d_pool_re)

        mlp = mlp_work(self.config['filters'])

        rq = mlp(q_pool_re)
        show_layer_info('MLP', rq)
        rd = mlp(d_pool_re)
        show_layer_info('MLP', rd)
        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
