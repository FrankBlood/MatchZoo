#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
data_loader
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 18-4-22下午3:47
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

import pandas as pd
import json
import random
import codecs
from pyltp import Segmentor
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

class Data_Loader(object):
    def __init__(self):
        self.excel_path = prodir + '/data/ZXSQA/zxs_pairs.xlsx'
        self.json_path = prodir + '/data/ZXSQA/zxs_pairs.json'
        self.arranged_json_path = prodir + '/data/ZXSQA/arrange_zxs_pairs.json'
        self.template_dict_path = prodir + '/data/ZXSQA/template_dict.json'

        self.train_path = prodir + '/data/ZXSQA/arrange_zxs_pairs.train.json'
        self.val_path = prodir + '/data/ZXSQA/arrange_zxs_pairs.val.json'
        self.test_path = prodir + '/data/ZXSQA/arrange_zxs_pairs.test.json'

        self.word_index_path = prodir + '/data/TKQA/word_index.json'

        self.train_size = 6666
        self.val_size = 814
        self.test_size = 823

        self.max_len = 200

        self.segmentor = Segmentor()
        self.segmentor.load(rootdir + "/LTP/ltp-data/ltp_data_v3.4.0/cws.model")


    def pairs2json(self, data_path):
        df = pd.read_excel(data_path)
        count = 0
        for index, line in df.iterrows():
            data = {'pattern': line['pattern'], 'template': line['template']}
            count += 1
            yield json.dumps(data, ensure_ascii=False)
        print(count)


    def get_templates_dict(self, data_path, save_path):
        df = pd.read_excel(data_path)
        count = 0
        templates_dict = {}
        for index, line in df.iterrows():
            template = line['template']
            if template in templates_dict:
                templates_dict[template] += 1
            else:
                templates_dict[template] = 1
                count += 1
        print("There are %d templates."%count)
        with open(save_path, 'w') as fw:
            json.dump(templates_dict, fw, ensure_ascii=False)


    def arrange_by_template(self, data_path):
        arranged_data = {}
        with codecs.open(data_path, encoding='utf8') as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                line = json.loads(line)
                if line['template'] in arranged_data:
                    arranged_data[line['template']].append(line)
                else:
                    arranged_data[line['template']] = [line]

        with open(self.arranged_json_path, 'w') as fw:
            json.dump(arranged_data, fw, ensure_ascii=False)
            print("Save successfully.")

        return arranged_data

    def split_data(self, data_path, rate=0.1):
        train_data = []
        val_data = []
        test_data = []
        with open(data_path, 'r') as fp:
            arranged_data = json.load(fp)
            for k, v in arranged_data.iteritems():
                if len(v) > 10:
                    random.shuffle(v)
                    train_data += v[:int(len(v)*(1-rate*2))]
                    val_data += v[int(len(v)*(1-rate*2)):int(len(v)*(1-rate*1))]
                    test_data += v[int(len(v)*(1-rate*1)):]
                else:
                    train_data += v

        print(len(train_data))
        print(len(val_data))
        print(len(test_data))

        with open(data_path[:-5]+'.train.json', 'w') as fw:
            for line in train_data:
                fw.write(json.dumps(line, ensure_ascii=False)+'\n')

        with open(data_path[:-5]+'.val.json', 'w') as fw:
            for line in val_data:
                fw.write(json.dumps(line, ensure_ascii=False)+'\n')

        with open(data_path[:-5]+'.test.json', 'w') as fw:
            for line in test_data:
                fw.write(json.dumps(line, ensure_ascii=False)+'\n')

    def tokenizer_prepare(self, data_path):
        texts = []
        with codecs.open(data_path, encoding='utf8') as fp:
            while True:
                if not fp.readline().strip():
                    break
                try:
                    line = json.loads(fp.readline().strip())
                    pattern = ' '.join(self.segmentor.segment(str(line['pattern'])))
                    template = ' '.join(self.segmentor.segment(str(line['template'])))
                    # print(pattern)
                    # print(template)
                    texts.append(pattern)
                    texts.append(template)
                except:
                    print("The wrong line is", line)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        # for k,v in tokenizer.word_counts.iteritems():
        #     print(k, v)
        print('size of word counts:', len(tokenizer.word_counts))
        print('size of word index:', len(tokenizer.word_index))
        self.tokenizer = tokenizer

    def save_word_index(self):
        with open(self.word_index_path, 'w') as fw:
            json.dump(self.tokenizer.word_index, fw)
            print("Save successfully.")


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

                pattern = str(line['pattern'])
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

                for n_template in self.nagative_sampling(template, kb, sample_size=sample_size):
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

    def nagative_sampling(self, template, kb, sample_size):
        key_list = kb.keys()
        key_list.remove(template)
        return random.sample(key_list, sample_size)


def run_pairs2json():
    data_loader = Data_Loader()
    print("save data to direct path.")
    with open(data_loader.json_path, 'w') as fw:
        for data in data_loader.pairs2json(data_loader.excel_path):
            fw.write(data + '\n')

def run_get_templates_dict():
    data_loader = Data_Loader()
    data_loader.get_templates_dict(data_loader.excel_path,
                                   data_loader.template_dict_path)

def run_arrange_by_template():
    data_loader = Data_Loader()
    a = data_loader.arrange_by_template(data_loader.json_path)
    print(a)
    print(len(a))

def run_split_data():
    data_loader = Data_Loader()
    data_loader.split_data(data_loader.arranged_json_path, rate=0.1)

def run_tokenizer_prepare():
    data_loader = Data_Loader()
    data_loader.tokenizer_prepare(data_loader.train_path)
    data_loader.save_word_index()

def run_data_generator():
    data_loader = Data_Loader()
    with open(data_loader.word_index_path, 'r') as fp:
        word_index = json.load(fp)
    with open(data_loader.template_dict_path, 'r') as fp:
        kb = json.load(fp)

    for i, j in data_loader.tk_data_generator(data_loader.train_path, word_index, kb, batch_size=100):
        # print(i, j)
        print()

if __name__ == "__main__":
    # run_pairs2json()
    # run_get_templates_dict()
    # run_arrange_by_template()
    run_split_data()
    # run_tokenizer_prepare()
    # run_data_generator()