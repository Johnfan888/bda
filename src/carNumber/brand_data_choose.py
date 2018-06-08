# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:53:40 2018

@author: dell-1
"""

import pandas as pd
import numpy as np
data = pd.read_csv('E:\\盐城数据\\data\\train_20171215.txt',sep='\t')
train_A = pd.read_csv('E:\\盐城数据\\data\\train_A.txt', )

all_data = data.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'cnt':np.sum})
all_data = all_data[all_data['date']<1032]

all_data = pd.concat([all_data,train_A],axis=0)
all_data = all_data.reset_index(drop=True)

# 时间特征
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
train_df = all_data_1.iloc[:1031,:] 
#test_df = all_data_1.iloc[1031:,:]
train_df = train_df.drop(['cnt'],axis=1)
#train = pd.concat([data,train_df],axis=0)
train = data.merge(train_df,on=['date','day_of_week'])




brand1 = train[train['brand']==1]
brand2 = train[train['brand']==2]
brand3 = train[train['brand']==3]
brand4 = train[train['brand']==4]
brand5 = train[train['brand']==5]

brand1.to_csv('brand1.csv',index=None)
brand2.to_csv('brand2.csv',index=None)
brand3.to_csv('brand3.csv',index=None)
brand4.to_csv('brand4.csv',index=None)
brand5.to_csv('brand5.csv',index=None)