#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
tk_data
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 18-4-11下午3:35
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

import pandas as pd
import json
import random
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from matplotlib import pyplot as plt
from pyltp import Segmentor
import codecs
random.seed(1000)

class TK_Data(object):
    def __init__(self):
        self.excel_path = prodir+'/data/TKQA/tk_triples.xlsx'
        self.json_path = prodir+'/data/TKQA/tk_triples.json'
        self.arranged_json_path = prodir+'/data/TKQA/tk_triples.arranged.json'
        self.word_index_path = prodir+'/data/TKQA/word_index.json'
        self.train_data_path = prodir+'/data/TKQA/tk_triples.arranged.train.json'
        self.val_data_path = prodir+'/data/TKQA/tk_triples.arranged.val.json'
        self.tess_data_path = prodir+'/data/TKQA/tk_triples.arranged.test.json'
        self.kb_path = prodir+'/data/TKQA/primaryTemplate.json'
        self.segmentor = Segmentor()
        self.segmentor.load(rootdir+"/LTP/ltp-data/ltp_data_v3.4.0/cws.model")


    def xls2json(self, data_path):
        df = pd.read_excel(data_path)
        count = 0
        for index, line in df.iterrows():
            if line['primaryQuestion'] == 0:
                count += 1
                print(count)
                data = {'primaryQuestion': line['pattern_id'], 'pattern': line['pattern'],
                        'template': line['template']}
            else:
                data = {'primaryQuestion': line['primaryQuestion'], 'pattern': line['pattern'],
                        'template': line['template']}
            yield json.dumps(data, ensure_ascii=False)


    def get_primary_question(self, data_path, save_path):
        df = pd.read_excel(data_path)
        count = 0
        fw = open(save_path, 'w')
        for index, line in df.iterrows():
            if line['primaryQuestion'] == 0:
                fw.write(line['pattern']+'\n')
                count+=1
        print(count)
        fw.close()


    def arrange_by_primaryquestion(self, data_path):
        arranged_data = {}
        with codecs.open(data_path, encoding='utf8') as fp:
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                line = json.loads(line)
                if line['primaryQuestion'] in arranged_data:
                    arranged_data[line['primaryQuestion']].append(line)
                else:
                    arranged_data[line['primaryQuestion']] = [line]

        with open(self.arranged_json_path, 'w') as fw:
            json.dump(arranged_data, fw)
            print("Save successfully.")

        return arranged_data


    def split_arranged_data(self, data_path, rate=0.1):
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


    def split_data(self, data_path, rate=0.1):
        with open(data_path, 'r') as fp:
            all_lines = fp.readlines()
            random.shuffle(all_lines)
            data_len = len(all_lines)
            train_data = all_lines[: int(data_len*(1-rate*2))]
            val_data = all_lines[int(data_len*(1-rate*2)): int(data_len*(1-rate*1))]
            test_data = all_lines[int(data_len*(1-rate*1)):]

        with open(data_path[:-5]+'.train.json', 'w') as fw:
            for line in train_data:
                fw.write(line)

        with open(data_path[:-5]+'.val.json', 'w') as fw:
            for line in val_data:
                fw.write(line)

        with open(data_path[:-5]+'.test.json', 'w') as fw:
            for line in test_data:
                fw.write(line)


    def stat_data(self, data_path):
        primaryQuestion = []
        pattern = []
        template = []
        with open(data_path, 'r') as fp:
            for line in fp.readlines():
                line = json.loads(line.strip())
                primaryQuestion.append(line['primaryQuestion'])
                pattern.append(line['pattern'])
                template.append(line['template'])
        stat_dict = {}
        for i in primaryQuestion:
            if i in stat_dict:
                stat_dict[i] += 1
            else:
                stat_dict[i] = 1
        num = 0
        all_count = []
        for k,v in stat_dict.iteritems():
            num += v
            all_count.append(v)

        plt.violinplot(all_count,
                       showmeans=True,
                       showmedians=False,
                       showextrema=True)
        plt.show()

        print("The stat dict of the primary question:", stat_dict)
        print("The num of all samples:", num)
        print("The num of primary question:", len(stat_dict))


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


    def char_tokenizer_prepare(self, data_path):
        texts = []
        with codecs.open(data_path, encoding='utf8') as fp:
            while True:
                if not fp.readline().strip():
                    break
                try:
                    line = json.loads(fp.readline().strip())
                    pattern = ' '.join([char.strip() for char in line['pattern']])
                    template = ' '.join([char.strip() for char in line['template']])
                    # print(pattern)
                    # print(template)
                    texts.append(pattern)
                    texts.append(template)
                except:
                    print("The wrong line is", line)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        for k,v in tokenizer.word_counts.iteritems():
            print(k, v)
        print('size of word counts:', len(tokenizer.word_counts))
        print('size of word index:', len(tokenizer.word_index))
        self.char_tokenizer = tokenizer


    def save_word_index(self):
        with open(self.word_index_path, 'w') as fw:
            json.dump(self.tokenizer.word_index, fw)
            print("Save successfully.")


    def get_kb(self, save_path):
        kb = {}
        count = 0
        num = 0
        with open(self.train_data_path, 'r') as fp:
            while True:
                line = fp.readline()
                num += 1
                if not line:
                    print("There are %d lines." % num)
                    print("Same answer is %s." % count)
                    with open(save_path, 'w') as fw:
                        json.dump(kb, fw)
                    return kb
                else:
                    line = json.loads(line.strip())
                    if line['primaryQuestion'] not in kb:
                        kb[line['primaryQuestion']] = [line['template']]
                    elif line['template'] not in kb[line['primaryQuestion']]:
                        kb[line['primaryQuestion']].append(line['template'])
                        print(str(line['primaryQuestion']))
                        count += 1
                        print(line['template'], 'is a new answer for the question.')
                    else:
                        continue


def run_xls2json():
    tk_data = TK_Data()
    print("save data to direct path.")
    with open(tk_data.json_path, 'w') as fw:
        for data in tk_data.xls2json(tk_data.excel_path):
            fw.write(data + '\n')


def run_get_primary_question():
    tk_data = TK_Data()
    tk_data.get_primary_question(tk_data.excel_path, prodir+'/data/TKQA/primaryQuestion.txt')


def run_stat_data():
    tk_data = TK_Data()
    tk_data.stat_data(tk_data.json_path)


def run_split_data():
    tk_data = TK_Data()
    # tk_data.split_data(tk_data.json_path, rate=0.1)
    tk_data.split_arranged_data(tk_data.arranged_json_path, rate=0.1)


def run_tokenizer_prepare():
    tk_data = TK_Data()
    # tk_data.tokenizer_prepare(tk_data.train_data_path)
    tk_data.char_tokenizer_prepare(tk_data.train_data_path)
    # tk_data.save_word_index()

def run_arrange_by_primaryquestion():
    tk_data = TK_Data()
    a = tk_data.arrange_by_primaryquestion(tk_data.json_path)
    print(a)
    print(len(a))

def run_get_kb():
    tk_data = TK_Data()
    tk_data.get_kb(tk_data.kb_path)

def func():
    pass


if __name__ == "__main__":
    # run_xls2json()
    # run_stat_data()
    # run_split_data()
    run_tokenizer_prepare()
    # run_arrange_by_primaryquestion()
    # run_get_kb()
    run_get_primary_question()