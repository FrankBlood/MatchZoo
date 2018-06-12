#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Models
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 18-4-13下午2:44
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:3])
PRO_NAME = 'MatchZoo'
prodir = rootdir + '/Research/' + PRO_NAME
print(prodir)
sys.path.insert(0, prodir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from inputs.data_loader import Data_Loader

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D, GlobalAveragePooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.layers import Dropout, Input, Activation, Flatten, Reshape
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Merge
from keras.layers.merge import concatenate, add, dot, multiply
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import numpy as np
import json

class Networks(object):
    def __init__(self, embedding_dim=200, num_rnn=200, num_dense=200,
                 rate_dropout_rnn=0.5, rate_dropout_dense=0.5, act='relu'):
        self.embedding_dim = embedding_dim
        self.num_rnn = num_rnn
        self.num_dense = num_dense
        self.rate_dropout_rnn = rate_dropout_rnn
        self.rate_dropout_dense = rate_dropout_dense
        self.act = act

    def info(self, data_loader):
        with open(data_loader.word_index_path, 'r') as fp:
            self.word_index = json.load(fp)
        self.nb_words = len(self.word_index) + 1
        self.question_length = data_loader.max_len
        self.utterance_length = data_loader.max_len


    def ELSTMAttention(self):
        print("Building the ELSTMAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(LSTM(units=self.num_rnn,
                                          dropout=self.rate_dropout_rnn,
                                          recurrent_dropout=self.rate_dropout_rnn,
                                          return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        # question_last_state = rnn(embedded_sequences_question)
        # utterance_last_state = rnn(embedded_sequences_utterance)
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention = concatenate([question_last_state, utterance_last_state])
        attention = Dense(self.num_rnn*2, activation='tanh')(attention)

        attention_question = dot([question, attention], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_question = Activation('softmax')(attention_question)

        attention_utterance = dot([utterance, attention], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'ELSTMAttention', model

    def EAttention(self):
        print("Building the EAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(GRU(units=self.num_rnn,
                                         dropout=self.rate_dropout_rnn,
                                         recurrent_dropout=self.rate_dropout_rnn,
                                         return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        # question_last_state = rnn(embedded_sequences_question)
        # utterance_last_state = rnn(embedded_sequences_utterance)
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention = concatenate([question_last_state, utterance_last_state])
        attention = Dense(self.num_rnn*2, activation='tanh')(attention)

        attention_question = dot([question, attention], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_question = Activation('softmax')(attention_question)

        attention_utterance = dot([utterance, attention], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'EAttention', model


    def BiLSTM(self):
        print("Building the BiLSTM...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])


        # 定义只保留最后一层的RNN网络
        rnn = Bidirectional(LSTM(units=self.num_rnn,
                                 dropout=self.rate_dropout_rnn,
                                 recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn(embedded_sequences_question)
        utterance = rnn(embedded_sequences_utterance)

        merged = concatenate([question, utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'BiLSTM', model

    def BiGRU(self):
        print("Building the BiGRU...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])


        # 定义只保留最后一层的RNN网络
        rnn = Bidirectional(GRU(units=self.num_rnn,
                                dropout=self.rate_dropout_rnn,
                                recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn(embedded_sequences_question)
        utterance = rnn(embedded_sequences_utterance)

        merged = concatenate([question, utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'BiGRU', model

    def ESelfLSTMAttention(self):
        print("Building the ESelfLSTMAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(LSTM(units=self.num_rnn,
                                          dropout=self.rate_dropout_rnn,
                                          recurrent_dropout=self.rate_dropout_rnn,
                                          return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention_question = Dense(self.num_rnn*2, activation='tanh')(question_last_state)
        attention_question = dot([question, attention_question], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_question = Activation('softmax')(attention_question)

        attention_utterance = Dense(self.num_rnn*2, activation='tanh')(utterance_last_state)
        attention_utterance = dot([utterance, attention_utterance], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'ESelfLSTMAttention', model

    def ESelfAttention(self):
        print("Building the ESelfAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(GRU(units=self.num_rnn,
                                         dropout=self.rate_dropout_rnn,
                                         recurrent_dropout=self.rate_dropout_rnn,
                                         return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention_question = Dense(self.num_rnn*2, activation='tanh')(question_last_state)
        attention_question = dot([question, attention_question], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_question = Activation('softmax')(attention_question)

        attention_utterance = Dense(self.num_rnn*2, activation='tanh')(utterance_last_state)
        attention_utterance = dot([utterance, attention_utterance], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'ESelfAttention', model

    def ECrossAttention(self):
        print("Building the ECrossAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(GRU(units=self.num_rnn,
                                         dropout=self.rate_dropout_rnn,
                                         recurrent_dropout=self.rate_dropout_rnn,
                                         return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention_q = Dense(self.num_rnn*2, activation='tanh')(question_last_state)
        attention_u = Dense(self.num_rnn * 2, activation='tanh')(utterance_last_state)

        attention_question = dot([question, attention_u], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_question = Activation('softmax')(attention_question)

        attention_utterance = dot([utterance, attention_q], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense*2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
                      loss='categorical_crossentropy',
                      # loss='binary_crossentropy',
                      optimizer='nadam',
                      # optimizer='rmsprop',
                      metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'ESelfAttention', model

    def ECrossLSTMAttention(self):
        print("Building the ECrossLSTMAttention...")
        embedding = Embedding(input_dim=self.nb_words,
                              output_dim=self.embedding_dim,
                              embeddings_initializer='random_uniform')

        def get_last_state(bidirection):
            return concatenate([bidirection[:, -1, :self.num_rnn], bidirection[:, 0, self.num_rnn:]])

        # 定义带每个step隐藏的RNN网络
        rnn_with_seq = Bidirectional(LSTM(units=self.num_rnn,
                                          dropout=self.rate_dropout_rnn,
                                          recurrent_dropout=self.rate_dropout_rnn,
                                          return_sequences=True))

        # 定义只保留最后一层的RNN网络
        # rnn = Bidirectional(CuDNNLSTM(units=self.num_rnn,
        #                     dropout=self.rate_dropout_rnn,
        #                     recurrent_dropout=self.rate_dropout_rnn))

        # 输入层和词嵌入层
        sequence_question = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_question = embedding(sequence_question)
        sequence_utterance = Input(shape=(self.utterance_length,), dtype='int32')
        embedded_sequences_utterance = embedding(sequence_utterance)

        # 计算带每个step隐藏层的RNN输出
        question = rnn_with_seq(embedded_sequences_question)
        utterance = rnn_with_seq(embedded_sequences_utterance)

        # 计算只保留最后一层的RNN网络
        question_last_state = Lambda(lambda x: get_last_state(x))(question)
        utterance_last_state = Lambda(lambda x: get_last_state(x))(utterance)

        # 计算注意力机制的权重
        attention_q = Dense(self.num_rnn * 2, activation='tanh')(question_last_state)
        attention_u = Dense(self.num_rnn * 2, activation='tanh')(utterance_last_state)

        attention_question = dot([question, attention_u], axes=-1)
        # attention_question = Lambda(lambda x: K.exp(x))(attention_question)
        attention_utterance = dot([utterance, attention_q], axes=-1)
        # attention_utterance = Lambda(lambda x: K.exp(x))(attention_utterance)

        attention_question = Activation('softmax')(attention_question)
        attention_utterance = Activation('softmax')(attention_utterance)

        new_question = dot([question, attention_question], axes=1)
        new_utterance = dot([utterance, attention_utterance], axes=1)

        # # 对有hidden state的输出用attention加权求和
        # question = Permute([2, 1])(question)
        # utterance = Permute([2, 1])(utterance)
        #
        # new_question = multiply([question, attention_question])
        # new_utterance = multiply([utterance, attention_utterance])
        # new_question = Permute([2, 1])(new_question)
        # new_utterance = Permute([2, 1])(new_utterance)
        #
        # new_question = Lambda(lambda x: K.sum(x, axis=1))(new_question)
        # new_utterance = Lambda(lambda x: K.sum(x, axis=1))(new_utterance)

        # merged = concatenate([question, utterance])
        merged = concatenate([new_question, new_utterance])
        merged = Dense(self.num_dense * 2, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(self.num_dense, activation=self.act)(merged)
        merged = Dropout(self.rate_dropout_dense)(merged)
        merged = BatchNormalization()(merged)

        preds = Dense(2, activation='softmax')(merged)
        # preds = Dense(1, activation='sigmoid')(merged)

        ########################################
        ## train the model
        ########################################
        model = Model(inputs=[sequence_question, sequence_utterance], outputs=preds)
        model.compile(
            loss='categorical_crossentropy',
            # loss='binary_crossentropy',
            optimizer='nadam',
            # optimizer='rmsprop',
            metrics=['acc'])
        model.summary()
        # print(STAMP)
        return 'ESelfAttention', model


def main():
    network = Networks()
    data_loader = Data_Loader()
    network.info(data_loader)
    # network.ELSTMAttention()
    # network.EAttention()
    # network.BiLSTM()
    # network.BiGRU()
    # network.ESelfLSTMAttention()
    network.ECrossLSTMAttention()
    # network.ESelfAttention()


def func():
    pass


if __name__ == "__main__":
    main()