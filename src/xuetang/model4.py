#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import tree
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def get_train_data():
    train_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_train.csv')
#   过滤掉男女未知和血糖最大的女性
    train_data = train_data[(True-train_data['id'].isin([580]))]
    train_data = train_data[(True-train_data['xtang'].isin([train_data['xtang'].max()]))]
#   对年龄归一化
    train_data['age'] = train_data['age']/100.0
#将均值填充在缺失值中
    train_drop = train_data.drop(['id','sex','age','time',
                            #  'c2','c_sum','y4','y3','hj','c1',
                            # 'dcell','xb','y1','sscell','hd', 'hl','b1','lcell','zcell',
                                 'xtang'],axis=1)
    tianchong = train_drop.mean()
    train_fill = train_drop.fillna(tianchong)
#对数值型进行归一化
    x_max = train_fill.max(axis=0)
    x_min = train_fill.min(axis=0)
    x1 = (train_fill-x_min)/(x_max-x_min)
# #与另外几列合并
    train_final = pd.concat([train_data['id'],train_data['sex'],train_data['age'],
                             x1,train_data['xtang']],axis=1)
    return x_max,x_min,tianchong,train_final
def get_test_data():
    test_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_test.csv')
    #   对年龄归一化
    test_data['age'] = test_data['age']/100.0
    #将均值填充在缺失值中
    test_drop = test_data.drop(['id','sex','age','time'],axis=1)
    x_max, x_min,tianchong,_ = get_train_data()
    test_fill = test_drop.fillna(tianchong)
    #对数值型进行归一化
    x1 = (test_fill-x_min)/(x_max-x_min)
    #与另外几列合并
    test_final = pd.concat([test_data['id'],test_data['sex'],test_data['age'],
                             x1],axis=1)
    # print test_final
    return test_final
def get_label(train_data,test_data,judge):
    if judge == 0:
        train_data['unique'] = map(lambda x:1 if x <= 4.6 else 0, train_data['xtang'])
    elif judge == 1:
        train_data['unique'] = map(lambda x:1 if x > 7 else 0, train_data['xtang'])

    X_train = train_data.drop(['id','sex','xtang','unique'],axis=1)
    y_train = train_data.loc[:,'unique']

    X_new = X_train[['gz','age']]
    clf = ensemble.RandomForestClassifier(n_estimators=200).fit(X_new,y_train)
    a = cross_validation.cross_val_score(clf, X_new, y_train)
    print a.mean()
    X_test = test_data[['gz','age']]
    unique1 = clf.predict(X_test)
    unique2 = clf.predict(X_test)
    unique3 = clf.predict(X_test)
    unique4 = clf.predict(X_test)
    unique5 = clf.predict(X_test)
    unique6 = clf.predict(X_test)
    unique7 = clf.predict(X_test)
    unique8 = clf.predict(X_test)
    unique9 = clf.predict(X_test)

    out = []
    for i in range(0,1000):
        a = unique1[i]+unique2[i]+unique3[i]+unique4[i]+unique5[i]+unique6[i]+unique7[i]+unique8[i]+unique9[i]
        # print a
        if a >= 5:
            out.append(1)
        else:
            out.append(0)
    test_data['unique'] = out
    test1 = test_data[test_data['unique']==1]
    return test1

def get_linear(train_data,test_data,alpha1):
    X_train = train_data.drop(['id', 'sex','xtang'], axis=1)
    y_train = train_data.loc[:, 'xtang']
    X_test = test_data.drop(['id', 'sex','unique'], axis=1)
    clf1 = linear_model.Ridge(alpha=alpha1)
    clf1.fit(X_train,y_train)
    y_pred = clf1.predict(X_test)

    res = pd.DataFrame({
        'id':test_data['id'],
        'y_pred': y_pred
    })
    return res
def get_svr(train_data,test_data):
    X_train = train_data.drop(['id', 'xtang',], axis=1)
    y_train = train_data.loc[:, 'xtang']
    X_test = test_data.drop(['id' , 'unique'], axis=1)
    clf1 = SVR(kernel='rbf')
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    res = pd.DataFrame({
        'id': test_data['id'],
        'y_pred': y_pred
    })
    return res

_,_,_,train_data = get_train_data()
# 将train数据分为train_small,train_big,train_mediun
train_small = train_data[train_data['xtang']<=4.6]
train_big = train_data[train_data['xtang']>7]
train_medium = train_data[(train_data['xtang']>4.6)&(train_data['xtang']<=7)]
# 将train_medium数据分为男，女
train_man = train_medium[train_medium['sex'] == 1]
train_woman = train_medium[train_medium['sex'] == 0]

# 将test_data数据打上标签，分为test_small,test_big,test_medium
test_data = get_test_data()
test_small = get_label(train_data,test_data,0)
test_big = get_label(train_data,test_data,1)
# print len(test_small)
# print len(test_big)
# 找出test_smalll 和test_big的id号，删去,得到中间的数据
id_del = []
for i in test_small['id']:
    id_del.append(i)
for j in test_big['id']:
    id_del.append(j)
test_medium = test_data[(True-test_data['id'].isin(id_del))]
# print len(test_medium)
# 将中间的数据划分为男，女
test_man = test_medium[test_medium['sex'] == 1]
test_woman = test_medium[test_medium['sex'] == 0]

# 训练small和big 数据
res_small = get_svr(train_small,test_small)
res_big = get_svr(train_big,test_big)

# 分开训练train_medium的男，女
res_man = get_linear(train_man,test_man,alpha1=0.3)
res_woman = get_linear(train_woman,test_woman,alpha1=0.1)

res_sum = pd.concat([res_small,res_man,res_woman,res_big],axis=0)
res_sum = res_sum.sort_values(['id'], ascending='False')
res_sum = res_sum.reset_index(drop=True)
print res_sum
res_sum['y_pred'].to_csv('/root/word2vec/tangniaobing/tang/pre/pre_1_7.csv',index=None)