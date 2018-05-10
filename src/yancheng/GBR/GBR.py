# -*- coding: utf-8 -*-
# 2018 年　1 月　31　日
# 使用特征：date, day_of_week ,n_time
# n_time :　这周五据上周五隔了多少天
# 线上MSE: 868215.07
# 线下测试80万


import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
# train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
# train_data.columns = ['date', 'day_of_week', 'brand', 'cnt']
# train_data = train_data.drop(['brand'],axis=1)
# train_dispose = train_data[['date','day_of_week', 'cnt']].\
#                 groupby(['date','day_of_week']).agg('sum').reset_index()
# print train_dispose.loc[91]
# train_1 = train_dispose[train_dispose['day_of_week']==1]
# train_2 = train_dispose[train_dispose['day_of_week']==2]
# train_3 = train_dispose[train_dispose['day_of_week']==3]
# train_4 = train_dispose[train_dispose['day_of_week']==4]
# train_5 = train_dispose[train_dispose['day_of_week']==5]
# train_6 = train_dispose[train_dispose['day_of_week']==6]
# train_7 = train_dispose[train_dispose['day_of_week']==7]
# # print train_1
# std_1 = train_1.loc[:,'cnt'].values.std()
# std_2 = train_2.loc[:,'cnt'].values.std()
# std_3= train_3.loc[:,'cnt'].values.std()
# std_4= train_4.loc[:,'cnt'].values.std()
# std_5= train_5.loc[:,'cnt'].values.std()
# std_6= train_6.loc[:,'cnt'].values.std()
# std_7= train_7.loc[:,'cnt'].values.std()
#
#
# mean_1 = train_1.loc[:,'cnt'].values.mean()
# mean_2 = train_2.loc[:,'cnt'].values.mean()
# mean_3 = train_3.loc[:,'cnt'].values.mean()
# mean_4 = train_4.loc[:,'cnt'].values.mean()
# mean_5 = train_5.loc[:,'cnt'].values.mean()
# mean_6 = train_6.loc[:,'cnt'].values.mean()
# mean_7 = train_7.loc[:,'cnt'].values.mean()
# print std_1/mean_1
# print std_2/mean_2
# print std_3/mean_3
# print std_4/mean_4
# print std_5/mean_5
# print std_6/mean_6
# print std_7/mean_7

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
test_data = pd.read_csv('/root/word2vec/yancheng/data/test_A.txt', sep='\t')
test_data = test_data[test_data['date'] >1032]

actions1 = train_data.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})
all_data = pd.concat([actions1,test_data],axis=0)
all_data = all_data.reset_index(drop=True)
# 周一
def get_train_part(day_number):
    train_1 = all_data[all_data['day_of_week']==day_number]
    index_1 = train_1.index
    count_1 = [0,]
    for i in range(0,len(train_1)-1):
        count_1.append(index_1[i+1] - index_1[i])
    train_1['n_time'] = count_1
    return train_1
train_1 = get_train_part(1)
train_2 = get_train_part(2)
train_3 = get_train_part(3)
train_4 = get_train_part(4)
train_5 = get_train_part(5)
train_6 = get_train_part(6)
train_7 = get_train_part(7)

all_data_1 = pd.concat([train_1,train_2,train_3,train_4,train_5,train_6,train_7],axis=0)
all_data_1 = all_data_1.sort_index()
train_df = all_data_1.loc[:1031,:]

test_df = all_data_1.loc[1032:,:]

def submit():
    X_train = train_df.drop(['count1'],axis = 1).values
    y_train = train_df['count1'].values
    X_test = test_df.drop(['count1'],axis = 1).values

    print "GradientBoostingRegressor"
    gbdt = GradientBoostingRegressor().fit(X_train, y_train)
    result1 = gbdt.predict(X_test)
    pre = []
    for i in result1:
        pre.append(abs(round(i)))

    out = pd.DataFrame({
        'date':test_data['date'],
        'gbdt_pre':pre
    })
    print out
    # out.to_csv('/root/word2vec/yancheng/data/pre/GBR130.csv',index=None,header=None,sep='\t')

def validation():
    # # 切分数据（训练集和测试集）
    df_train_target = train_df['count1'].values
    df_train_data = train_df.drop(['count1'],axis = 1).values
    cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=5, test_size=0.2,random_state=0)

    print "GradientBoostingRegressor"
    for train, test in cv:
        gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
        result1 = gbdt.predict(df_train_data[test])
        print(mean_squared_error(result1, df_train_target[test]))
        print '......'

validation()