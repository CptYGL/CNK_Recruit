# -*- coding: utf-8 -*-
# @author YGL
import pickle,json,os,re

regex_non_char = lambda line : re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9]').sub('',line['text'])
# lambda 函数,取json对象的text部分,去除中英文数字之外的东西


#创建或读取词id映射的函数
def create_vocab(in_fname, special=['<PAD>','<UNK>']):
    word2id = {}
    id2word = {}
    with open(in_fname,encoding='utf-8') as f:
        for line in f:
            # 中文字符串,只保留中英数字,分成单个字符
            wds = list(regex_non_char(json.loads(line)))
            for wd in wds:
                if wd not in word2id:
                    #这里id从2开始，0留作填充<padding>用，1留作<unknow>用
                    #到时可以从batch中计算mark，因为0表示了填充位置
                    word2id[wd] = len(word2id) + len(special)
            for id,spe in enumerate(special):
                word2id[spe] = id
    id2word = {it[1]: it[0] for it in word2id.items()}
    return (word2id, id2word)

def load_vocab(fname):
    word2id = {}
    with open(fname,'rb') as f:
        word2id = pickle.load(f)
    id2word = {it[1]: it[0] for it in word2id.items()}
    return (word2id, id2word)

#测试
word2i_fname = './corpus/word2i.vocab'
train_data_fname = './corpus/summary.jsonl'
if not os.path.isfile(word2i_fname):
    w2i, i2w =  create_vocab(train_data_fname )
    print(len(w2i))
    with open(word2i_fname,'wb') as f: 
        pickle.dump(w2i, f)
else:
    w2i, i2w = load_vocab(word2i_fname)
    print(len(w2i))#)
