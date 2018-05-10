#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def get_train_data():
    train_data = pd.read_csv('/root/word2vec/tangniaobing/tang/d_train.csv')
#   过滤掉男女未知和血糖最大的女性
    train_data = train_data[(True-train_data['id'].isin([580]))]
    train_data = train_data[(True-train_data['xtang'].isin([train_data['xtang'].max()]))]
#   对年龄归一化
    train_data['age'] = train_data['age']/100.0
# 将均值填充在缺失值中
    train_drop = train_data.drop(['id','sex','age','time','xtang'],axis=1)
    tianchong = train_drop.mean()
    train_fill = train_drop.fillna(tianchong)
#对数值型进行归一化
    x_max = train_fill.max(axis=0)
    x_min = train_fill.min(axis=0)
    x1 = (train_fill-x_min)/(x_max-x_min)

#与另外几列合并
    train_final = pd.concat([train_data['id'],train_data['sex'],train_data['age'],
                             x1,train_data['xtang']],axis=1)
    train_man = train_final[train_final['sex'] == 1]
    train_woman = train_final[train_final['sex'] == 0]
    return train_man
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('0.5mse', score, False)
def my_lgb(train,test):
    X_train = train.drop(['id', 'sex', 'xtang'], axis=1)
    y_train = train.loc[:, 'xtang']
    X_test = test.drop(['id', 'sex', 'xtang'], axis=1)
    y_test = test.loc[:, 'xtang']
    # 为lightgbm创建数据
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.7,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
        }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=3000,
                valid_sets=lgb_eval,
                verbose_eval=100,
                feval=evalerror,
                early_stopping_rounds=100)
    print('Start predicting...')
    y_pre3 = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    a = len(test)
    b = mean_squared_error(y_test,y_pre3)
    return a,b

def get_score(train,test,alpha):
    X_train = train.drop(['id', 'sex', 'xtang'], axis=1)
    y_train = train.loc[:, 'xtang']
    X_test = test.drop(['id', 'sex', 'xtang'], axis=1)
    y_test = test.loc[:, 'xtang']
    clf1 = linear_model.Ridge(alpha).fit(X_train,y_train)
    y_pre1 = clf1.predict(X_test)
    res_1 = test
    res_1['y_pre1'] = y_pre1
    return y_pre1,res_1
def my_random(train,test,):
    X_train = train.drop(['id', 'sex', 'xtang',], axis=1)
    y_train = train.loc[:, 'xtang']
    X_test = test.drop(['id', 'sex', 'xtang',], axis=1)
    y_test = test.loc[:, 'xtang']
    clf2 = RandomForestRegressor(n_estimators=1000,
                                    max_depth=7,
                                    max_features=0.2,
                                  max_leaf_nodes=100).fit(X_train,y_train)
    y_pre2 = clf2.predict(X_test)
    a = len(test)
    b = mean_squared_error(y_test,y_pre2)
    return a,b
train_man_all = get_train_data()
# 将数据七三分
train_man,test_man = train_test_split(train_man_all,test_size=0.3,random_state=0)
# 将结果用model预测
_,res1= get_score(train_man,test_man,alpha=0.3)
# 用y_pre1的结果划分>7和<=7，分开处理
test_small = res1[res1['y_pre1']<=7]
test_big = res1[res1['y_pre1']>7]
# 去掉y_pre
test_small = test_small.drop(['y_pre1'],axis=1)
test_big = test_big.drop(['y_pre1'],axis=1)
# print test_small
# # 预测
a1,b1 = my_random(train_man,test_small)
a2,b2 = my_lgb(train_man,test_big)

print b1/2
print b2/2
print (a1*b1+a2*b2)/(2*(a1+a2))


