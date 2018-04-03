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

class BiMPM(BasicModel):
    def __init__(self, config):
        super(BiMPM, self).__init__(config)
        self.__name = 'BiMPM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[BiMPM] parameter check wrong')
        print('[BiMPM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        # 获得BiRNN的最后step的hidden state
        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.config['hidden_size']], bidirection[:, 0, self.config['hidden_size']:]])

        # 第一种匹配方式
        def full_matching(context, last_state):
            # context = BatchNormalization()(context)
            # last_state = BatchNormalization()(last_state)
            matching = multiply([context, last_state])
            matching = Conv1D(filters=20, kernel_size=1, activation='tanh')(matching)
            return matching

        def full_matching_question(context):
            context_question, context_utterance = context[0], context[1]
            last_state_of_utterance = get_last_state(context_utterance)

            matching_question = full_matching(context_question, last_state_of_utterance)
            return matching_question

        def full_matching_utterance(context):
            context_question, context_utterance = context[0], context[1]
            last_state_of_question = get_last_state(context_question)

            matching_utterance = full_matching(context_utterance, last_state_of_question)
            return matching_utterance

        # Context Representation Layer: 定义带每个step隐藏的RNN网络（用GRU代替LSTM）
        context_representation = Bidirectional(GRU(units=self.config['hidden_size'],
                                                   dropout=self.config['dropout_rate'],
                                                   recurrent_dropout=self.config['dropout_rate'],
                                                   return_sequences=True))

        # Aggregation Layer: 定义只保留最后一层的RNN网络（用GRU代替LSTM）
        aggregation = Bidirectional(GRU(units=20,
                                        dropout=self.config['dropout_rate'],
                                        recurrent_dropout=self.config['dropout_rate']))

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        # 计算带每个step隐藏层的RNN输出
        context_question = context_representation(q_embed)
        show_layer_info('Context of query', context_question)
        context_utterance = context_representation(d_embed)
        show_layer_info('Context of doc', context_utterance)

        # 计算多角度matching向量
        matching_question = Lambda(lambda x: full_matching_question(x))([context_question, context_utterance])
        show_layer_info('Matching of query', matching_question)
        matching_utterance = Lambda(lambda x: full_matching_utterance(x))([context_question, context_utterance])
        show_layer_info('Matching of doc', matching_utterance)

        # 计算集合
        aggregation_question = aggregation(matching_question)
        show_layer_info('Aggregation of query', aggregation_question)
        aggregation_utterance = aggregation(matching_utterance)
        show_layer_info('Aggregation of doc', aggregation_utterance)

        # 计算分类评分
        merged = concatenate([aggregation_question, aggregation_utterance])
        merged = Dense(20 * 2, activation='relu')(merged)
        merged = Dropout(self.config['dropout_rate'])(merged)
        merged = BatchNormalization()(merged)
        show_layer_info('Merged 1', merged)

        merged = Dense(20, activation='relu')(merged)
        merged = Dropout(self.config['dropout_rate'])(merged)
        merged = BatchNormalization()(merged)
        show_layer_info('Merged 2', merged)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(merged)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(merged)
        show_layer_info('Dense', out_)

        # out_ = Dot(axes= [1, 1], normalize=True)([new_q_rep, new_d_rep])

        model = Model(inputs=[query, doc], outputs=out_)
        model.summary()
        return model