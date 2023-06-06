# -*- coding: utf-8 -*-
#@author:YGL
import re

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