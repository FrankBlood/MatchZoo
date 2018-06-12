#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
data_loader
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 18-4-11下午7:11
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:3])
print(rootdir)
PRO_NAME = 'MatchZoo'
prodir = rootdir + '/Research/' + PRO_NAME
print(prodir)
sys.path.insert(0, prodir)


if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import codecs
import random
import json
import re
from pyltp import Segmentor
from keras.preprocessing.sequence import pad_sequences


class Data_Loader(object):
    def __init__(self):
        self.train_data_path = prodir+'/data/TKQA/tk_triples.arranged.train.json'
        self.val_data_path = prodir+'/data/TKQA/tk_triples.arranged.val.json'
        self.tess_data_path = prodir+'/data/TKQA/tk_triples.arranged.test.json'
        self.word_index_path = prodir+'/data/TKQA/word_index.json'
        self.kb_path = prodir+'/data/TKQA/primaryTemplate.json'
        self.max_len = 100
        self.train_size = 131315
        self.val_size = 16238
        self.test_size = 16513
        self.segmentor = Segmentor()
        self.segmentor.load(rootdir+"/LTP/ltp-data/ltp_data_v3.4.0/cws.model")


    def tk_data_generator(self, data_path, word_index, kb, batch_size, sample_size=2):

        with codecs.open(data_path, encoding='utf8') as fp:
            all_lines = fp.readlines()
            random.shuffle(all_lines)
            idx = 0
            count = 0
            batch_q, batch_a, batch_l = [], [], []
            while True:
                if idx == len(all_lines):
                    idx = 0

                line = all_lines[idx]
                line = json.loads(line.strip(), encoding='utf8')

                pattern = line['pattern']
                pattern_seq = re.split('\s', ' '.join(self.segmentor.segment(str(pattern.decode('utf8')))))
                pattern_id = []
                for word in pattern_seq:
                    try:
                        pattern_id.append(word_index[word.decode('utf8')])
                    except:
                        pass

                template = line['template']
                template_seq = re.split('\s', ' '.join(self.segmentor.segment(str(template.decode('utf8')))))
                template_id = []
                for word in template_seq:
                    try:
                        template_id.append(word_index[word.decode('utf8')])
                    except:
                        pass

                batch_q.append(pattern_id)
                batch_a.append(template_id)
                batch_l.append([1, 0])
                # batch_l.append([1])
                count += 1

                # print('pos question', pattern)
                # print('pos answer', template)
                # print('pos question', ' '.join(pattern_seq))
                # print('pos answer', ' '.join(template_seq))
                # print(pattern_id)
                # print(template_id)
                # print('#'*10)

                primary_question = str(line['primaryQuestion'])

                for n_template in self.nagative_sampling(primary_question, kb, sample_size=sample_size):
                    n_template_seq = re.split('\s', ' '.join(self.segmentor.segment(str(n_template.decode('utf8')))))
                    n_template_id = []
                    for word in n_template_seq:
                        try:
                            n_template_id.append(word_index[word.decode('utf8')])
                        except:
                            pass

                    batch_q.append(pattern_id)
                    batch_a.append(n_template_id)
                    batch_l.append([0, 1])
                    # batch_l.append([0])

                    # print('neg question', pattern)
                    # print('neg answer', n_template)
                    # print('pos question', ' '.join(pattern_seq))
                    # print('pos answer', ' '.join(n_template_seq))
                    # print(pattern_id)
                    # print(n_template_id)
                    # print('#' * 10)

                if count == batch_size:
                    # print('idx is', idx)
                    yield ([pad_sequences(batch_q, maxlen=self.max_len),
                            pad_sequences(batch_a, maxlen=self.max_len)],
                            batch_l)
                    batch_q, batch_a, batch_l = [], [], []
                    count = 0

                idx += 1

    def nagative_sampling(self, primaryQuestion, kb, sample_size):
        key_list = kb.keys()
        key_list.remove(primaryQuestion)
        N_templates = []
        for q in random.sample(key_list, sample_size):
            N_templates.append(random.sample(kb[q], 1)[0])
        return N_templates



def cws():
    seg = Segmentor()
    seg.load("/home/irlab0/LTP/ltp-data/ltp_data_v3.4.0/cws.model")
    a = '你好，我想问一下，一般医疗是指的轻症么'
    print('\t'.join(seg.segment(a)))

def run_data_generator():
    data_loader = Data_Loader()
    with open(data_loader.word_index_path, 'r') as fp:
        word_index = json.load(fp)
    with open(data_loader.kb_path, 'r') as fp:
        kb = json.load(fp)

    for i, j in data_loader.tk_data_generator(data_loader.train_data_path, word_index, kb, batch_size=100):
        print(i, j)


def func():
    pass


if __name__ == "__main__":
    # cws()
    run_data_generator()