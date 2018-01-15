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
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60,90]:
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

def data_process(data):
    # 对数据进行归一化处理，并分成训练集和测试机
    # input：train_dispose
    # output: train_data1 , test_data1
    data = data.drop(['1day_std'],axis=1)

    # 将day_of_week转化为它的平均值
    def flag_to_number(data):
        d = {1: 2364, 2: 2495, 3: 2094, 4: 1625, 5: 1889, 6: 412, 7: 742}
        if d.has_key(data):
            return d[data]
        else:
            return -1
    data['day_mean'] = data['day_of_week'].map(flag_to_number)
    data['cnt_log'] = data['cnt'].apply(lambda x: np.log(1 + x))
    data.to_csv('/root/word2vec/yancheng/data/train_dispose2.csv',index=None)
    data_all = data.drop(['date', 'day_of_week', 'cnt','cnt_log'], axis=1)
    # 对数值型进行归一化
    scaler = StandardScaler().fit(data_all)
    data_uniform = scaler.transform(data_all)
    data_frame = pd.DataFrame(data_uniform)

    data_frame = pd.concat([data['date'],data['day_of_week'],data['cnt'],
                              data['cnt_log'],data_frame],axis=1)

    train_data1 = data_frame[data_frame['date']<=822]
    test_data1 = data_frame[data_frame['date']>822]

    return train_data1,test_data1

def many_model(train,test):

    X_train1 = train.drop(['date','cnt','cnt_log'],axis=1)
    y_train1 = train.loc[:,'cnt_log']
    X_test1 = test.drop(['date','cnt','cnt_log'],axis=1)
    y_test1 = test.loc[:,'cnt']
    model1 = linear_model.LinearRegression().fit(X_train1,y_train1)

    res = pd.DataFrame({
        'date': test['date'],
    })
    res['y_pre_log'] = model1.predict(X_test1)
    res['y_pre'] = res['y_pre_log'].apply(lambda x: np.exp(x) - 1)
    a = mean_squared_error(y_test1, res['y_pre'])

    # res['diff'] = res['cnt'] - res['y_pred']
    # out = pd.merge(test,res,on=['date','cnt_log'],how='outer')
    # out.to_csv('/root/word2vec/yancheng/data/pre1.csv',index=None)
    return a , res

frame_boss = feature_1(90)
for i in range(91,len(train_dispose)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
# 获得全部数据的前x天的矩阵
frame_boss =  (frame_boss.T).reset_index(drop=True)
train_data1,test_data1 = data_process(frame_boss)
a , out = many_model(train_data1,test_data1)
print a
print out









