# -*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
train_data.columns = ['date', 'day_of_week', 'brand', 'cnt']
train_data = train_data.drop(['brand'],axis=1)
train_dispose0 = train_data[['date','day_of_week', 'cnt']].\
                groupby(['date','day_of_week']).agg('sum').reset_index()
def flag_to_number(data):
    d = {1: 2364, 2: 2495, 3: 2094, 4: 1625, 5: 1889, 6: 412, 7: 742}
    if d.has_key(data):
        return d[data]
    else:
        return -1
train_dispose0['day_mean'] = train_dispose0['day_of_week'].map(flag_to_number)
train_dispose = train_dispose0[train_dispose0['date'] <=832]
test_dispose = train_dispose0[train_dispose0['date'] >=832].reset_index()
y_test = test_dispose['cnt']
#test_data是test的x值
test_data = test_dispose[['date','day_of_week','day_mean']]
def date_range(step,end):
    if end-step < 0:
        print 'date range not fit....'
    else:
        a = end-step
        b = end-1
        return a,b
def feature_1(date_end):
    frame_final = train_dispose.iloc[date_end,:]
    begin,end = date_range(7,date_end)
    frame = train_dispose[(train_dispose.index >= begin) & (train_dispose.index<= end)]
    frame_final = pd.concat([frame_final,frame['cnt'],],axis=0)
    return frame_final

def mylgb(train_data):
    # create dataset for lightgbm
    X_train1 = train_data.drop(['date','cnt',], axis=1)
    y_train1 = train_data.loc[:, 'cnt']
    model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=10000
    ).fit(X_train1,y_train1,)
    return model,X_train1

frame_boss = feature_1(7).reset_index(drop=True)
for i in range(8,len(train_dispose)):
    frame_final = feature_1(i)
    frame_final = (frame_final.T).reset_index(drop=True)
    # print frame_final
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
frame_boss = frame_boss.T.reset_index(drop=True)
frame_boss.columns = ['date','day_of_week','day_mean','cnt',
                      'd1','d2','d3','d4','d5','d6','d7',]
# print frame_boss
mymodel3,X_train1 = mylgb(frame_boss)
test_data1 = test_data

y_lables = train_dispose.loc[:,'cnt'].to_frame(name='cnt')
# #先将第1032天的数据加上，不用预测
y_pre = ["1032\t1306",]
y_pre1 = [2051,]
for j in range( 1, len(test_data)):
    length_y = len(y_lables)
    frame_final1 = test_data.loc[j]
    x_list = []
    begin,end = date_range(7,length_y)
    frame = y_lables[(y_lables.index >= begin) & (y_lables.index<= end)]
    frame_final = pd.concat([ frame_final1,frame['cnt'],], axis=0).reset_index(drop=True)
    frame_final = frame_final.iloc[1:]
    print frame_final
    date = int(frame_final1[0])
    a = np.reshape(frame_final.as_matrix(),(1,-1))

    b = abs(round(mymodel3.predict(a)))
    c = str(date)+'\t'+str(int(b))
    y_pre.append(c)
    y_pre1.append(b)
    y_lables.loc[length_y] = b

sum = 0.0
for i in range(0,len(y_pre1)):
    e = (y_pre1[i]-y_test[i])**2
    sum+=e
print sum/len(y_pre1)



