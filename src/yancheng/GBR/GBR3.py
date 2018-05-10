# -*- coding: utf-8 -*-
# 添加特征，


import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
all_data = train_data.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})

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

def flag_to_number1(data):
    # 方差
    d = {1:989,2:999,3:854,4:878,5:874,6:597,7:1425}
    if d.has_key(data):
        return d[data]
    else:
        return -1
# all_data_1['day_std'] = all_data_1['day_of_week'].map(flag_to_number1)

# 去除前30天

# 前一天的min
def date_range(step,end):
    if end-step < 0:
        print 'date range not fit....'
    else:
        a = end-step
        b = end-1
        return a,b
def feature_1(date_end):
    frame_final = all_data_1.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7, 14, 21, 30,]:
        begin,end = date_range(day,date_end)
        frame = all_data_1[(all_data_1.index >= begin) & (all_data_1.index<= end)]
        # 将取出的数据累加
        frame1 = frame['count1'].agg(
            {
              str(day) + 'day_max': np.max,
              str(day) + 'day_std': np.std,
             })
        frame_final = pd.concat([frame_final, frame1], axis=0)
    return frame_final

frame_boss = feature_1(31)
for i in range(32,len(all_data_1)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
frame_boss =  (frame_boss.T).reset_index(drop=True)
# print frame_boss
train_df1 = frame_boss.iloc[:581]
test_df1 = frame_boss.iloc[581:696,:]

train_df2 = frame_boss.iloc[:696]
test_df2 = frame_boss.iloc[672:835,:]

train_df3 = frame_boss.iloc[:835]
test_df3 = frame_boss.iloc[835:,:]


def validation(train_df,test_df):
    # # 切分数据（训练集和测试集）

    X_train = train_df[['date','day_of_week','n_time','7day_std','3day_max']].values
    y_train = train_df['count1'].values
    X_test = test_df[['date', 'day_of_week', 'n_time','7day_std','3day_max']].values
    y_test = test_df['count1'].values

    gbdt = GradientBoostingRegressor().fit(X_train,y_train)
    result1 = gbdt.predict(X_test)
    print (mean_squared_error(result1, y_test))

validation(train_df1,test_df1)
validation(train_df2,test_df2)
validation(train_df3,test_df3)

