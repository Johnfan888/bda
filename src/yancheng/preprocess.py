#!/usr/bin/env python
# -- coding:utf-8 --
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
def get_brand_data(brand_value):
    train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
    train_data.columns = ['date', 'day_of_week', 'brand', 'cnt']
    brand_data = train_data[train_data['brand'] == brand_value]
    brand_data = brand_data[['date', 'day_of_week', 'brand', 'cnt']] \
        .groupby(['date', 'day_of_week', 'brand']).agg('sum').reset_index()
    return brand_data
brand_1 = get_brand_data(brand_value=1)

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
    frame_final = brand_1.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60]:
        # 获取终止日期前day天的数据
        begin,end = date_range(day,date_end)
        frame = brand_1[(brand_1.index >= begin) & (brand_1.index<= end)]
        # 将取出的数据累加
        frame1 = frame['cnt'].agg(
            { str(day) + 'day_avg': np.mean,
              str(day) + 'day_std': np.std,
              str(day) + 'day_min': np.min,
             })

        frame_final = pd.concat([frame_final,frame1],axis=0)
    return frame_final
def get_score(train_data):
    X_train = train_data.drop(['date','cnt','1day_std'],axis=1)
    y_train = train_data.loc[:,'cnt']
    # print y_train
    clf1 = linear_model.LinearRegression()
    a = cross_validation.cross_val_score(clf1, X_train, y_train, scoring='neg_mean_squared_error').mean()
    print a

frame_boss = feature_1(60)
for i in range(61,len(brand_1)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
# 获得全部数据的前x天的矩阵
frame_boss =  (frame_boss.T).reset_index(drop=True)
# print frame_boss

get_score(frame_boss)







