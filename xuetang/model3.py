#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import linear_model
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
    #去掉一些不好的属性
    train_final = train_final.drop(['b_sum', 'c1', 'jg', 'y1', 'y2', 'y3', 'y4', 'y5', 'h_count',
                               'xhb', 'hj', 'dcell', 'sscell', 'sjcell'], axis=1)

    train_man = train_final[train_final['sex']==1]
    train_woman = train_final[train_final['sex']==0]
    return train_man,train_woman,x_max,x_min,tianchong


def get_test_data():
    test_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_test.csv')
    #   对年龄归一化
    test_data['age'] = test_data['age']/100.0
    #将均值填充在缺失值中
    test_drop = test_data.drop(['id','sex','age','time'],axis=1)
    _, _, x_max, x_min,tianchong = get_train_data()
    test_fill = test_drop.fillna(tianchong)
    #对数值型进行归一化
    x1 = (test_fill-x_min)/(x_max-x_min)
    #与另外几列合并
    test_final = pd.concat([test_data['id'],test_data['sex'],test_data['age'],
                             x1],axis=1)
    test_final = test_final.drop(['b_sum', 'c1', 'jg', 'y1', 'y2', 'y3', 'y4', 'y5', 'h_count',
                                    'xhb', 'hj', 'dcell', 'sscell', 'sjcell'], axis=1)
    test_man = test_final[test_final['sex']==1]
    test_woman = test_final[test_final['sex']==0]
    return test_man,test_woman

def get_linear(train_data,test_data,alpha1):
    X_train = train_data.drop(['id', 'sex', 'xtang'], axis=1)
    y_train = train_data.loc[:, 'xtang']
    X_test = test_data.drop(['id', 'sex',], axis=1)
    clf1 = linear_model.Ridge(alpha=alpha1)
    clf1.fit(X_train,y_train)
    y_pred = clf1.predict(X_test)

    res = pd.DataFrame({
        'id':test_data['id'],
        'y_pred': y_pred
    })
    return res

train_man,train_woman,_,_,_ = get_train_data()
test_man,test_woman = get_test_data()

res_man = get_linear(train_man,test_man,alpha1=0.3)
res_woman = get_linear(train_man,test_woman,alpha1=0.1)

res_sum = pd.concat([res_man,res_woman],axis=0)
res_sum = res_sum.sort_values(['id'], ascending='False')
res_sum = res_sum.reset_index(drop=True)
res_sum['y_pred'].to_csv('/root/word2vec/tangniaobing/tang/pre/pre_1_6.csv',index=None)
# print res_sum
