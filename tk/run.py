#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
train
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 18-4-13下午2:54
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICECS"] = '1'

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


from keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import json

from inputs.data_loader import Data_Loader
from networks import Networks


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def train(data_loader, network):

    # model_name, model = network.ELSTMAttention()
    # model_name, model = network.EAttention()

    # model_name, model = network.BiLSTM()
    # model_name, model = network.BiGRU()

    # model_name, model = network.ESelfLSTMAttention()
    # model_name, model = network.ESelfAttention()

    # model_name, model = network.ECrossLSTMAttention()
    model_name, model = network.ECrossAttention()

    with open(data_loader.word_index_path, 'r') as fp:
        word_index = json.load(fp)

    with open(data_loader.kb_path, 'r') as fp:
        kb = json.load(fp)

    sample_size = 1
    batch_size = 100/(1+sample_size)

    # previous_model_path = curdir+'/models/basic_baselineWed_Nov_29_03:54:35_2017_95000_update_kdb_ch.h5'
    # if os.path.exists(previous_model_path):
    #     print("previous_model_path:", previous_model_path)
    #     model.load_weights(previous_model_path)

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
    bst_model_path = './models/' + model_name +now_time + '_' +str(data_loader.train_size) +'.h5'
    print('bst_model_path:', bst_model_path)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_weights_only=True, save_best_only=True)

    for i in range(2):
        if os.path.exists(bst_model_path):
            model.load_weights(bst_model_path)

        print("This is the %d loop."%i)
        model.fit_generator(data_loader.tk_data_generator(data_loader.train_data_path, word_index=word_index, kb=kb, batch_size=batch_size, sample_size=sample_size),
                            steps_per_epoch=data_loader.train_size/batch_size+1,
                            # steps_per_epoch=20,
                            validation_data=data_loader.tk_data_generator(data_loader.val_data_path, word_index=word_index, kb=kb, batch_size=batch_size, sample_size=sample_size),
                            validation_steps=data_loader.val_size/batch_size+1,
                            # validation_steps=1,
                            epochs=15,
                            shuffle=True,
                            callbacks=[model_checkpoint])

        if os.path.exists(bst_model_path):
            model.load_weights(bst_model_path)

        loss, acc = model.evaluate_generator(data_loader.tk_data_generator(data_loader.tess_data_path,
                                                               word_index=word_index,
                                                               kb=kb,
                                                               batch_size=batch_size,
                                                               sample_size=1),
                                 steps=data_loader.test_size / batch_size + 1)
        print('test loss and acc is ', loss, acc)


def evaluate(data_loader, network, model_path):
    model_name, model = network.ELSTMAttention()
    # model_name, model = network.BiLSTM()

    with open(data_loader.word_index_path, 'r') as fp:
        word_index = json.load(fp)

    with open(data_loader.kb_path, 'r') as fp:
        kb = json.load(fp)

    batch_size = 100/(1+1)

    model.load_weights(model_path)
    print(data_loader.test_size)
    loss, acc = model.evaluate_generator(data_loader.tk_data_generator(data_loader.tess_data_path,
                                                           word_index=word_index,
                                                           kb=kb,
                                                           batch_size=batch_size,
                                                           sample_size=1),
                             steps=data_loader.test_size/batch_size+1)
    print('test loss and acc is ', loss, acc)


def main():

    data_loader = Data_Loader()
    network = Networks()
    network.info(data_loader)
    train(data_loader, network)


def run_evaluate():
    data_loader = Data_Loader()
    network = Networks()
    network.info(data_loader)
    # model_path = './models/ELSTMAttentionSat_Apr_14_15:26:18_2018_131315.h5'
    model_path = './models/ELSTMAttentionSat_Apr_14_01:02:09_2018_131315.h5'
    evaluate(data_loader, network, model_path)


if __name__ == "__main__":
    main()
    # run_evaluate()