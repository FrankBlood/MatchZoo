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

class ECautiousBiMPM(BasicModel):
    def __init__(self, config):
        super(ECautiousBiMPM, self).__init__(config)
        self.__name = 'ECautiousBiMPM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size',
                   'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[ECautiousBiMPM] parameter check wrong')
        print('[ECautiousBiMPM] init done', end='\n')

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

        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        conv1d = Conv1D(filters=self.config['hidden_size'], kernel_size=3, padding='valid', activation='relu')
        convert = Dense(self.config['embed_size'], activation='tanh')
        transfer = Conv1D(filters=self.config['hidden_size'], kernel_size=1, padding='same', activation='tanh')
        extract = Conv1D(filters=self.config['hidden_size'], kernel_size=1, padding='same', activation='relu')
        rnn_with_seq = Bidirectional(GRU(units=self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                         recurrent_dropout=self.config['dropout_rate'], return_sequences=True))

        # Context Representation Layer: 定义带每个step隐藏的RNN网络（用GRU代替LSTM）
        context_representation = Bidirectional(GRU(units=self.config['hidden_size'],
                                                   dropout=self.config['dropout_rate'],
                                                   recurrent_dropout=self.config['dropout_rate'],
                                                   return_sequences=True))

        # Aggregation Layer: 定义只保留最后一层的RNN网络（用GRU代替LSTM）
        aggregation = Bidirectional(GRU(units=20,
                                        dropout=self.config['dropout_rate'],
                                        recurrent_dropout=self.config['dropout_rate']))

        q_conv = conv1d(q_embed)
        q_conv = Dropout(self.config['dropout_rate'])(q_conv)
        show_layer_info("Conv1D Q", q_conv)
        q_global_pool = GlobalMaxPooling1D()(q_conv)
        q_global_pool = convert(q_global_pool)
        show_layer_info("Global Max Pooling Q", q_global_pool)
        q_global_pool_repeat = RepeatVector(self.config['text1_maxlen'])(q_global_pool)
        show_layer_info("Repeat Global Max Pooling Q", q_global_pool_repeat)
        merge_embed_conv_q = concatenate([q_embed, q_global_pool_repeat])
        merge_embed_conv_q = transfer(merge_embed_conv_q)
        show_layer_info("Merge Embed and Conv of Q", merge_embed_conv_q)

        d_conv = conv1d(d_embed)
        d_conv = Dropout(self.config['dropout_rate'])(d_conv)
        show_layer_info("Conv1D D", d_conv)
        d_global_pool = GlobalMaxPooling1D()(d_conv)
        d_global_pool = convert(d_global_pool)
        show_layer_info("Global Max Pooling Q", d_global_pool)
        d_global_pool_repeat = RepeatVector(self.config['text2_maxlen'])(d_global_pool)
        show_layer_info("Repeat Global Max Pooling D", d_global_pool_repeat)
        merge_embed_conv_d = concatenate([d_embed, d_global_pool_repeat])
        merge_embed_conv_d = transfer(merge_embed_conv_d)
        show_layer_info("Merge Embed and Conv of D", merge_embed_conv_d)

        # 计算带每个step隐藏层的RNN输出
        context_q = context_representation(merge_embed_conv_q)
        show_layer_info('Context of query', context_q)
        context_d = context_representation(merge_embed_conv_d)
        show_layer_info('Context of doc', context_d)

        # 计算多角度matching向量
        matching_q = Lambda(lambda x: full_matching_question(x))([context_q, context_d])
        show_layer_info('Matching of query', matching_q)
        matching_d = Lambda(lambda x: full_matching_utterance(x))([context_q, context_d])
        show_layer_info('Matching of doc', matching_d)

        # 计算集合
        aggregation_question = aggregation(matching_q)
        show_layer_info('Aggregation of query', aggregation_question)
        aggregation_utterance = aggregation(matching_d)
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
