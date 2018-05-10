#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd

a2 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man1.csv')
a3 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man2.csv')
a4 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man3.csv')
a5 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man4.csv')
a6 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man5.csv')



a = pd.concat([a2,a3,a4,a5,a6],axis=1)



b2 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man6.csv')
b3 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man7.csv')
b4 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man8.csv')
b5 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man9.csv')
b6 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man10.csv')


b = pd.concat([b2,b3,b4,b5,b6,],axis=1)

# print b

c2 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man11.csv')
c3 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man12.csv')
c4 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man13.csv')
c5 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man14.csv')
c6 = pd.read_csv('/root/word2vec/tangniaobing/tang/mse_man15.csv')


c = pd.concat([c2,c3,c4,c5,c6,],axis=1)

# print c
d = pd.concat([a2['min_index'],a3['min_index'],a4['min_index'],a5['min_index'],a6['min_index'],
               b2['min_index'],b3['min_index'],b4['min_index'],b5['min_index'],b6['min_index'],
               c2['min_index'],c3['min_index'],c4['min_index'],c5['min_index'],c6['min_index'],
                 ],axis=1)
print d
# d.to_csv('/root/word2vec/tangniaobing/tang/count_mode.csv',index=None)