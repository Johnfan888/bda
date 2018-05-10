# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:27:19 2018

@author: wen
"""
#!/usr/bin/env python
# -- coding:utf-8 --
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
train_data.columns = ['date', 'day_of_week', 'brand', 'cnt']
train_data = train_data.drop(['brand'],axis=1)
train_dispose = train_data[['date','day_of_week', 'cnt']].\
                groupby(['date','day_of_week']).agg('sum').reset_index()
# print train_dispose
#读入并处理test数据
test_data = pd.read_csv('/root/word2vec/yancheng/data/test.txt', sep='\t')
test_data.columns = ['date', 'day_of_week']
def date_range(step,end):
    # params: step:前step天，end:结束日期
    # return：起始日期，终止日期
    if end-step < 0:
        print 'date range not fit....'
    else:
        a = end-step
        b = end-1
        return a,b

def feature_1(date_end):
    # 获得终止日期前1，2，3，4，7，14，21，30，60天的列表
    #获取第61天的值，也就是y_real值
    frame_final = train_dispose.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60, 90]:
        # 获取终止日期前day天的数据
        begin,end = date_range(day,date_end)
        frame = train_dispose[(train_dispose.index >= begin) & (train_dispose.index<= end)]
        # 将取出的数据累加
        frame1 = frame['cnt'].agg(
            { str(day) + 'day_avg': np.mean,
              str(day) + 'day_std': np.std,
              str(day) + 'day_min': np.min,
             })

        frame_final = pd.concat([frame_final,frame1],axis=0)
    return frame_final
def flag_to_number(data):
    d = {1: 2364, 2: 2495, 3: 2094, 4: 1625, 5: 1889, 6: 412, 7: 742}
    if d.has_key(data):
        return d[data]
    else:
        return -1
def data_process(data):
    # 对数据进行归一化处理，并分成训练集和测试机
    # input：train_dispose
    # output: train_data1 , test_data1
    data = data.drop(['1day_std'],axis=1)
    # 将day_of_week转化为它的平均值
    data['day_mean'] = data['day_of_week'].map(flag_to_number)


    data_all = data.drop(['date', 'day_of_week', 'cnt','day_mean'], axis=1)
    # 对数值型进行归一化
#    scaler = StandardScaler().fit(data_all)
#    data_uniform = scaler.transform(data_all)


    data_frame = pd.concat([data['date'],data['day_of_week'],
                            data['day_mean'],
                           data_all, data['cnt']],axis=1)

#    train_data1 = data_frame[data_frame['date']<=822]
#    test_data1 = data_frame[data_frame['date']>822]
#    # 去掉周六周天
#    train_data1 = train_data1[(True-train_data1['day_of_week'].isin([7]))]
#    test_data1 = test_data1[(True-test_data1['day_of_week'].isin([7]))]
    return data_frame

def model(train_data):
    X_train1 = train_data.drop(['date','cnt',],axis=1)
    y_train1 = train_data.loc[:,'cnt']
    model1 = linear_model.LinearRegression().fit(X_train1,y_train1)
    return model1

frame_boss = feature_1(90)
for i in range(91,len(train_dispose)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
# 获得全部数据的前x天的矩阵
frame_boss =  (frame_boss.T).reset_index(drop=True)
data_frame = data_process(frame_boss)
print data_frame
mymodel = model(data_frame)
#a , out = many_model(train_data1,test_data1)
#print a
#print out
#将day_of_week改为day_mean
test_data['day_mean'] = test_data['day_of_week'].map(flag_to_number)
# test_data1 = test_data.drop(['day_of_week',],axis=1)
y_lables = train_dispose.loc[:,'cnt'].to_frame(name='cnt')

#y_list = []
#for i in range(0, len(y_lables)):
#    y_list.append(y_lables[i])
#先将第1032天的数据加上，不用预测
y_pre = [1306,]

length_y = len(y_lables)
frame_final1 = test_data.loc[1]
for day in [1, 2, 3, 4, 7, 14, 21, 30, 60, 90]:
# 获取终止日期前day天的数据
    begin,end = date_range(day,length_y)
    frame = y_lables[(y_lables.index >= begin) & (y_lables.index<= end)]
# 将取出的数据累加
    frame1 = frame['cnt'].agg( { str(day) + 'day_avg': np.mean,
              str(day) + 'day_std': np.std,
              str(day) + 'day_min': np.min,
             })
    frame_final1 = pd.concat([frame_final1,frame1],axis=0)
frame_final1 = frame_final1.drop(['1day_std'])
frame_final1 = frame_final1.drop(['date'])

a = np.reshape(frame_final1.as_matrix(),(1,-1))
b = mymodel.predict(a)
print b