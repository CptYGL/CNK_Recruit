# -*- coding: utf-8 -*-
#@author:YGL
import pickle,json,re,torch


# 语料处理
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

# #测试
# word2i_fname = './corpus/word2i.vocab'
# train_data_fname = './corpus/summary.jsonl'
# if not os.path.isfile(word2i_fname):
#     w2i, i2w =  create_vocab(train_data_fname )
#     print(len(w2i))
#     with open(word2i_fname,'wb') as f: 
#         pickle.dump(w2i, f)
# else:
#     w2i, i2w = load_vocab(word2i_fname)
#     print(len(w2i))#)


# 正则自定义函数
class regexer():
    '''
    正则规则:
    edubg 学历 : 一般为符号后跟一句如"1.xxxx学历"或者"xxx专业xxx学历"
    exp 经验 : 一般为"xx年以上xxx经验"而"x年工作经验"这种笼统的不算
    certi 证书 : 一般为"(有/拥有)xxx证书"
    使用方法:
    使用时判断sifind,为true则是有效返回;
    返回一个列表 [起始下标,结束下标,标签]
    '''
    def __init__(self, kw=''):
        self.kw = kw
        self._isfind = False
        self.edubg_pattern = re.compile(u'[\u4e00-\u9fa5]*学历|专业.*学历')
        self.exp_pattern = re.compile(u'\S年[\u4e00-\u9fa5]{5,}经验')
        self.certi_pattern = re.compile(u'获?[持|得|有][\u4E00-\u9FA5A-Za-z0-9]+证书?')
    def match(self, s):
        if self.kw == '学历':
            match = self.edubg_pattern.search(s)
        elif self.kw == '经验':
            match = self.exp_pattern.search(s)
        elif self.kw == '证书':
            match = self.certi_pattern.search(s)
        else:pass
        self._isfind = True if match else False
        if self._isfind:
            (b,e) = match.span()
            # print([b,e,self.kw])
            return [b,e,self.kw]

# 多分类交叉熵
def multicat_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

