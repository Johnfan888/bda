#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn import ensemble
import numpy as np

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
                             train_fill,train_data['xtang']],axis=1)
    train_man = train_final[train_final['sex']==1]
    train_woman = train_final[train_final['sex']==0]
    return train_man,train_woman,x_max,x_min,tianchong,train_final
def get_test_data():
    test_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_test.csv')
    #   对年龄归一化
    test_data['age'] = test_data['age']/100.0
    #将均值填充在缺失值中
    test_drop = test_data.drop(['id','sex','age','time'],axis=1)
    _, _, x_max, x_min,tianchong,_ = get_train_data()
    test_fill = test_drop.fillna(tianchong)
    #对数值型进行归一化
    x1 = (test_fill-x_min)/(x_max-x_min)
    #与另外几列合并
    test_final = pd.concat([test_data['id'],test_data['sex'],test_data['age'],
                             x1],axis=1)
    # print test_final
    test_man = test_final[test_final['sex']==1]
    test_woman = test_final[test_final['sex']==0]
    return test_man,test_woman,test_final

def classifier(train_data,test_data,unique_value):
    # 将train血糖值分割，返回两部分数据
    # 给test数据打上unique的标签
    train_data['unique'] = map(lambda x:1 if x>unique_value else 0, train_data['xtang'])
    print  np.count_nonzero(train_data['unique'])
    X_train = train_data.drop(['id','sex','age','xtang','unique'],axis=1)
    y_train = train_data.loc[:,'unique']
    X_test = test_data.drop(['id','age','sex'],axis=1)
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(X_train,y_train)
    # y_pred = clf.predict(X_test)
    # clf = neighbors.KNeighborsClassifier(5).fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    clf = ensemble.RandomForestClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print y_pred
    print np.count_nonzero(y_pred)
    test_data['unique'] = y_pred

    return train_data,test_data

def get_linear(train_data,test_data,alpha1):
    X_train = train_data.drop(['id', 'sex', 'xtang','unique'], axis=1)
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
    # 验证
    # a = cross_validation.cross_val_score(clf1, X_train, y_train, scoring='neg_mean_squared_error').mean()
    # train_sum = len(X_train)
    # return a,train_sum

def get_svr(train_data,test_data):
    X_train = train_data.drop(['id', 'sex', 'xtang','unique'], axis=1)
    y_train = train_data.loc[:, 'xtang']
    X_test = test_data.drop(['id', 'sex', 'unique'], axis=1)
    clf1 = SVR(kernel='rbf')
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    res = pd.DataFrame({
        'id': test_data['id'],
        'y_pred': y_pred
    })
    return res




train_man,train_woman,_,_,_,train_final = get_train_data()
test_man,test_woman,test_final = get_test_data()
# 将男性train数据打标签
# 将男性test数据通过分类得到标签
# train = pd.concat([train_man,train_woman],axis=0)
# test = pd.concat([test_man,test_woman],axis=0)
classifier(train_final,test_final,9)
# train_man,test_man =  classifier(train_man,test_man,9)

# 将男性数据按unique的不同分开训练
# train_man_1 = train_man[train_man['unique'] == 1]
# train_man_0 = train_man[train_man['unique'] == 0]
# test_man_1 = test_man[test_man['unique'] == 1]
# test_man_0 = test_man[test_man['unique'] == 0]
#
# # 训练数据
# res_man_1 = get_svr(train_man_1,test_man_1)
# res_man_0 = get_linear(train_man_0,test_man_0,alpha1=0.3)

# 按照上面的路给女性来一遍
# trian_woman,test_woman = classifier(train_woman,test_woman,9)
# train_woman_1 = trian_woman[trian_woman['unique'] == 1]
# train_woman_0 = trian_woman[trian_woman['unique'] == 0]
# test_woman_1 = test_woman[test_woman['unique'] == 1]
# test_woman_0 = test_woman[test_woman['unique'] == 0]
#
# res_woman_1 = get_svr(train_woman_1,test_woman_1)
# res_woman_0 = get_linear(train_woman_0,test_woman_0,alpha1=0.1)
# # 合并
# res_sum = pd.concat([res_man_1,res_woman_0,res_woman_1],axis=0)
# res_sum = res_sum.sort_values(['id'], ascending='False')
# res_sum = res_sum.reset_index(drop=True)
# res_sum['y_pred'].to_csv('/root/word2vec/tangniaobing/tang/pre/pre_1_2.csv',index=None)
# print res_sum





