# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:23:40 2018

@author: dell-1
最终版本：110名 / 699179
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
#train数据
train_data = pd.read_csv('E:\\盐城数据\\train_20171215.txt', sep='\t')
all_data = train_data.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})
all_data = all_data[all_data['date']<1032]
#A榜数据的合并
test_A = pd.read_csv('E:\\盐城数据\\test_A_20171225.txt',sep='\t')
answer_A = pd.read_csv('E:\\盐城数据\\answer_A_20180225.txt',sep='\t',header=None)
answer_A.columns = ['date','count1']
train_A = pd.merge(test_A,answer_A,on=['date'])
#将train数据和A榜数合并
all_data = pd.concat([all_data,train_A],axis=0)
all_data = all_data.reset_index(drop=True)

#B榜数据
test_B = pd.read_csv('E:\\盐城数据\\test_B_20171225.txt',sep='\t')
#将train数据，A榜，B榜数据结合起来
all_data = pd.concat([all_data,test_B],axis=0)
all_data = all_data.reset_index(drop=True)

# 时间特征,计算你n_time属性
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
#根据day_of_week赋值方差
def flag_to_number1(data):
    # 【0，1307】的方差
    d = {1:948, 2:1033, 3:874,4: 853,5: 850, 6:744, 7:1379}
    if data in d:
        return d[data]
    else:
        return -1     
all_data_1['day_std'] = all_data_1['day_of_week'].map(flag_to_number1)
#将全部数据划分为测试集合和submit集合
train_data = all_data_1.iloc[:1307,:] 
test_data = all_data_1.iloc[1307:,:]

def date_range(step,end):
#    a:初始date的index，b:末尾date的index
    if end-step < 0:
        print('date range not fit....')
    else:
        a = end-step
        b = end-1
        return a,b
  
def feature_1(date_end):
    frame_final = train_data.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7]:
        begin,end = date_range(day,date_end)
        frame = train_data[(train_data.index >= begin) & (train_data.index<= end)]
        # 将取出的数据累加
        frame1 = frame['count1'].agg(
            {
                str(day) + 'day_mean': np.mean,
             })
       
        frame_final = pd.concat([frame_final, frame1], axis=0)
    return frame_final

frame_boss = feature_1(7)
for i in range(8,len(train_data)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
#计算出所有train_data 的矩阵：
#count1,date,day_of_week,n_time,day_std,1day_mean,2day_mean,3day_mean,4day_mean,7day_mean
frame_boss =  (frame_boss.T).reset_index(drop=True)

def mymodel(train_data):
#    构建模型
    X_train = train_data.drop(['count1',],axis=1).values
    y_train = train_data['count1'].values
    model1 = GradientBoostingRegressor().fit(X_train,y_train)
    return model1
my_model = mymodel(frame_boss)


#submit数据处理
test_df = test_data.drop(['count1'],axis=1)
y_lables = train_data.loc[:,'count1'].to_frame(name='count1')
y_pre = []
for j in range(0, len(test_df)):
    length_y = len(y_lables)
    frame_final1 = test_df.iloc[j,:]
    for day in [1, 2, 3, 4, 7,]:
        begin,end = date_range(day,length_y)
        frame = y_lables[(y_lables.index >= begin) & (y_lables.index<= end)]
      
        frame1 = frame['count1'].agg(
            {
                str(day) + 'day_mean': np.mean,
             })
       
        frame_final1 = pd.concat([frame_final1,frame1],axis=0)

    a = np.reshape(frame_final1.as_matrix(), (1, -1))
    c = round(abs(my_model.predict(a)[0]))
    y_lables.ix[length_y] = pd.Series(c,index=y_lables.columns)
    y_pre.append(c)

res = pd.DataFrame({
    'date': test_data.loc[:,'date'],
    'res' : y_pre,
})
res.to_csv('E:\\盐城数据\\b2_1.csv',index=None,header=None,sep='\t')

