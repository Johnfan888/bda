#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import linear_model
from sklearn.feature_selection import f_regression
def get_train_data():
    train_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_train.csv')
#   过滤掉男女未知和血糖最大的女性
    train_data = train_data[(True-train_data['id'].isin([580]))]
    train_data = train_data[(True-train_data['xtang'].isin([train_data['xtang'].max()]))]
#   对年龄归一化
    train_data['age'] = train_data['age']/100.0
#将均值填充在缺失值中
    train_drop = train_data.drop(['id','sex','age','time',
                            #  'c2','c_sum','y4','y3','hj','c1'
                            #  'dcell','xb','y1','sscell','hd', 'hl','b1','lcell','zcell',
                                 'xtang'],axis=1)
    tianchong = train_drop.mean()
    train_fill = train_drop.fillna(tianchong)
#对数值型进行归一化
    x_max = train_fill.max(axis=0)
    x_min = train_fill.min(axis=0)
    x1 = (train_fill-x_min)/(x_max-x_min)
#与另外几列合并
    train_final = pd.concat([train_data['id'],train_data['sex'],train_data['age'],
                             x1,train_data['xtang']],axis=1)
    # train_final = train_final.drop(['b_sum', 'c1', 'jg', 'y1', 'y2', 'y3', 'y4', 'y5', 'h_count',
    #                            'xhb', 'hj', 'dcell', 'sscell', 'sjcell'], axis=1)
    # train_final = train_final.drop(['b1','b3','y1','y2','y3','y5','hj','xv','xvd',
    #                                 'zcell','lcell','dcell','sscell','sjcell'], axis=1)
    train_man = train_final[train_final['sex'] == 1]
    train_woman = train_final[train_final['sex'] == 0]
    return train_man,train_woman,train_final


def get_score(train_data,a):
    X_train = train_data.drop(['id', 'sex', 'xtang', ], axis=1)
    y_train = train_data.loc[:, 'xtang']
    # X_new = SelectKBest(f_regression, k=24).fit_transform(X_train, y_train)
    clf1 = linear_model.Ridge(alpha=a)
    a = cross_val_score(clf1, X_train, y_train, scoring='neg_mean_squared_error').mean()
    train_sum = len(X_train)

    return a,train_sum


train_man,train_woman,train_final = get_train_data()
train_man = train_final.drop(['b_sum', 'c1', 'jg', 'y1', 'y2', 'y3', 'y4', 'y5', 'h_count',
                               'xhb', 'hj', 'dcell', 'sscell', 'sjcell'], axis=1)
train_woman = train_woman.drop(['b1', 'b3', 'y1', 'y2', 'y3', 'y5', 'hj', 'xv', 'xvd',
                                 'zcell','lcell','dcell','sscell','sjcell'], axis=1)
a, man_sum = get_score(train_man, a=0.3)
b, woman_sum = get_score(train_woman, a=0.1)

print -a*0.5
print -b*0.5
print -(a * man_sum + b * woman_sum) / (2 * (man_sum + woman_sum))




