# -*- coding: utf-8 -*-
# author:YGL
# simple import
import numpy as np
import json,pickle
import torch,re
import torch.nn as nn
#internal import
from builder_vocab import load_vocab
# import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

bio = {'B':'1','I':'2'}
tag = {"职责": '01', "专业": '02', "知识": '03', 
            "素质": '04', "技能": '05', "程度": '06', 
            "福利": '07','学历':'08','经验':'09','证书':'10'}
label2idx = {'O':0}
for i in bio:
    for j in tag:
        label2idx[f'{i}-{j}']=int(bio[i]+tag[j])

#CRF-BIO数据集类
class CrfBIODataset(Dataset):
    def __init__(self, ori_fname, word2i, label2i, max_len=100):
        self.max_len = max_len
        self.datas = self.read_data(ori_fname, word2i, label2i)

    def read_data(self, fname, word2i, label2i):
        datas = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                (xil,yil) = self.tag_align(line,word2i,label2i)
                #排除空项,长度控制
                if not xil:
                    continue
                left = self.max_len - len(xil)
                if left > 0:
                    xil.extend([0] * left)
                    yil.extend([0] * left)
                else:
                    xil = xil[:self.max_len]
                    yil = yil[:self.max_len]
                    len(yil)
                datas.append((torch.tensor(xil), torch.tensor(yil)))
        return datas
    
    def tag_align(self,line,w2i,l2i):
        # 对单行进行对齐的逻辑部分
        line = json.loads(line)
        text = list(line['text'])
        labels = line['label']
        tags = ['O']*len(text)
        for label in labels:
            #潜在隐患,无法排除嵌套
            text[label[0]] = l2i[f'B-{label[2]}']
            for i in range((label[1]-label[0])):
                text[label[0]+i] = l2i[f'I-{label[2]}']
        # print(tags)
        xil,yil = [],[]
        for xi,yi in zip(text,tags):
            # .get( , 第二个参数表示默认int),w2i中1表示unknown,l2i中0表示"O"
            xil.append(w2i.get(xi,1))
            yil.append(l2i.get(yi,0))
        return (xil, yil)
    
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, index):
        return self.datas[index]
    
# 测试
# nerd = CrfBIODataset('./corpus/summary.jsonl', w2i, label2idx)
# nerdl = DataLoader(nerd, batch_size=3, shuffle=False)
# c = 0
# for x,y in nerdl:
#     print(x.shape)
#     print(x)
#     print(y.shape)
#     print(y)
#     #如果要用到mark，可以计算
#     mask = (x != 0)
#     print(mask)
#     c += 1
#     if c == 1:
#         break