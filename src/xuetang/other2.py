#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from dateutil.parser import parse
from sklearn import linear_model
from sklearn import cross_validation
train = pd.read_csv('/root/word2vec/tangniaobing/tang/d_train.csv')
test = pd.read_csv('/root/word2vec/tangniaobing/tang/d_test.csv')

def make_feat(train, test):

    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])
    data['time'] = (pd.to_datetime(data['time']) - parse('2017-10-09')).dt.days
    data.fillna(data.median(axis=0), inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat

def get_linear(train_data,test_data,alpha1):
    X_train = train_data.drop(['id', 'sex','xtang',], axis=1)
    y_train = train_data.loc[:, 'xtang']
    # print X_train
    X_test = test_data.drop(['id', 'sex','xtang'], axis=1)
    clf1 = linear_model.Ridge(alpha=alpha1)
    clf1.fit(X_train,y_train)
    # print X_test
    y_pred = clf1.predict(X_test)

    res = pd.DataFrame({
        'id':test_data['id'],
        'y_pred': y_pred
    })
    a = cross_validation.cross_val_score(clf1, X_train, y_train, scoring='neg_mean_squared_error').mean()
    # return res
    return a,len(X_train)



train_feat, test_feat = make_feat(train,test)
train_man = train_feat[train_feat['sex'] == 1]
train_woman = train_feat[train_feat['sex'] == 0]

test_man = test_feat[test_feat['sex'] == 1]
test_woman = test_feat[test_feat['sex'] == 0]

a,man_sum = get_linear(train_man,test_man,alpha1=0.3)
b,woman_sum = get_linear(train_woman,test_woman,alpha1=0.1)
print -a*0.5
print -b*0.5
print -(a * man_sum + b * woman_sum) / (2 * (man_sum + woman_sum))

#
# res_man = get_linear(train_man,test_man,alpha1=0.3)
# res_woman = get_linear(train_woman,test_woman,alpha1=0.1)
#
# res_sum = pd.concat([res_man,res_woman],axis=0)
# res_sum = res_sum.sort_values(['id'], ascending='False')
# res_sum = res_sum.reset_index(drop=True)
# res_sum['y_pred'].to_csv('/root/word2vec/tangniaobing/tang/pre/pre_1_9.csv',index=None)
# print res_sum