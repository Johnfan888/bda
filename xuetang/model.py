#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import cross_validation
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing

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
#与另外几列合并
    train_final = pd.concat([train_data['id'],train_data['sex'],train_data['age'],
                             x1,train_data['xtang']],axis=1)
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
    # print test_final
    test_man = test_final[test_final['sex']==1]
    test_woman = test_final[test_final['sex']==0]
    return test_man,test_woman

def linear(train_data,test_data,alpha1):
    X_train = train_data.iloc[:, 2:-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 2:]
    clf1 = linear_model.Ridge(alpha=alpha1)
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    result = pd.DataFrame({'id': test_data['id'],
                               'y_pred': y_pred,
                               })

    return result
def get_score(train_data,alpha1):
    X_train = train_data.iloc[:, 2:-1]
    y_train = train_data.iloc[:, -1]
    clf1 = linear_model.Ridge(alpha=alpha1)
    a = cross_validation.cross_val_score(clf1, X_train, y_train, scoring='neg_mean_squared_error').mean()
    train_sum = len(X_train)
    return a,train_sum

def run():
    train_man,train_woman,_,_,_ = get_train_data()
    test_man,test_woman = get_test_data()

    # 得出最终结果
    res_man = linear(train_man,test_man,alpha1=0.3)
    res_woman  = linear(train_woman,test_woman,alpha1=0.1)
    df = pd.concat([res_man, res_woman], axis=0)
    df = df.sort_values(['id'], ascending='False')
    df = df.reset_index(drop=True)
    print df
    a = []
    for i in df['y_pred']:
        a.append(round(i,3))
    df['y'] = a
    # df['y'].to_csv('/root/word2vec/tangniaobing/tang/pre/pre_lin1-1.csv', header=None,
    #                     index=None)
def cross_value():
    train_man, train_woman, _, _, _ = get_train_data()
    a, man_sum = get_score(train_man, alpha1=0.1)
    b, woman_sum = get_score(train_woman, alpha1=0.1)
    # print a / 2
    # print b / 2
    print (a * man_sum + b * woman_sum) / (2 * (man_sum + woman_sum))
    with open('/root/word2vec/tangniaobing/tang/zuhe.txt','a') as wr:
        wr.write(str(a/2)+' ')
        wr.write(str(b/2)+' ')
        wr.write(str((a * man_sum + b * woman_sum) / (2 * (man_sum + woman_sum))))
        wr.write('\n')
        # print (a/2),(b/2), (a * man_sum + b * woman_sum) / (2 * (man_sum + woman_sum))


cross_value()



