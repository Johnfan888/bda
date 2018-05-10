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
    return train_man,train_woman
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
    # predict
    y_pre3 = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    return y_pre3

def get_score(train,test,alpha):
    X_train = train.drop(['id', 'sex', 'xtang'], axis=1)
    y_train = train.loc[:, 'xtang']
    X_test = test.drop(['id', 'sex', 'xtang'], axis=1)
    y_test = train.loc[:, 'xtang'].as_matrix()
    clf1 = linear_model.Ridge(alpha).fit(X_train,y_train)
    clf2 = RandomForestRegressor(n_estimators=1000,
                                    max_depth=7,
                                    max_features=0.2,
                                  max_leaf_nodes=100).fit(X_train,y_train)
    y_pre1 = clf1.predict(X_test)
    y_pre2 = clf2.predict(X_test)
    return y_test,y_pre1,y_pre2

def mysort(y_test1,y_pre1,y_pre2,y_pre3):
    y_tuples = []
    for i in range(0,len(y_pre1)):
        a= y_test1[i]
        b = y_pre1[i]
        # c= (b-a)**2
        c = (a-b)
        d = y_pre2[i]
        # e = (a-d)**2
        e = (a-d)
        f = y_pre3[i]
        g = (a-f)
        # g = (a - f) ** 2
        # h = [a,b,c,d,e,f,g]
        h = [a,b,c,e,g]
        y_tuples.append(h)
    y_tuples1 = sorted(y_tuples, key=lambda x: x[2])
    return y_tuples1


def myMse(y_tuples1):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    Mse = []
    flag = int((y_tuples1[0][1]-3)*10)/3
    for i in range(0, len(y_tuples1)):
        e1 = y_tuples1[i][2]
        e2 = y_tuples1[i][4]
        e3 = y_tuples1[i][6]
        y1 = int((y_tuples1[i][1]-3)*10)/3
        if(y1==flag):
            sum1 += e1; sum2 += e2;sum3 +=e3
        else:
            mse = [sum1,sum2,sum3]; Mse.append(mse);  flag = y1
            sum1 = e1;  sum2 = e2; sum3=e3
    return Mse

train_man_all,train_woman_all = get_train_data()
train_man,test_man = train_test_split(train_man_all,test_size=0.3,random_state=0)
train_woman,test_woman = train_test_split(train_woman_all,test_size=0.3,random_state=0)

y_test1,man_pre1,man_pre2 = get_score(train_man,test_man,alpha=0.3)
y_test2,woman_pre1,woman_pre2 = get_score(train_woman,test_woman,alpha=0.1)
man_pre3 = my_lgb(train_man,test_man,)
woman_pre3 = my_lgb(train_man,test_man,)

y_man = mysort(y_test1,man_pre1,man_pre2,man_pre3)
y_woman = mysort(y_test2,woman_pre1,woman_pre2,woman_pre3)
man_diff = pd.DataFrame(y_man,columns=['y_test','y_pre1','e1','e2','e3'])
man_diff  = man_diff.sort_values(['y_pre1'], ascending='False')
woman_diff = pd.DataFrame(y_woman,columns=['y_test','y_pre1','e1','e2','e3'])
woman_diff  = woman_diff.sort_values(['y_pre1'], ascending='False')
# man_diff.to_csv('/root/word2vec/tangniaobing/tang/diff1.csv',index=None,header=None)
# woman_diff.to_csv('/root/word2vec/tangniaobing/tang/diff2.csv',index=None,header=None)

# mse_man = myMse(y_man)
# mse_woman = myMse(y_woman)
#
# mse1 = pd.DataFrame(mse_man)
# mse2 = pd.DataFrame(mse_woman)

# mse1.to_csv('/root/word2vec/tangniaobing/tang/mse1_1.csv',index=None,header=None)
# mse2.to_csv('/root/word2vec/tangniaobing/tang/mse2_1.csv',index=None,header=None)