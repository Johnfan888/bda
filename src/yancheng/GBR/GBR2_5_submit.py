# -*- coding: utf-8 -*-
# 提交结果
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
test_all = pd.read_csv('/root/word2vec/yancheng/data/test_A.txt', sep='\t')
test_data = test_all[test_all['date'] >1032]

actions1 = train_data.groupby(['date','day_of_week'], as_index=False)['cnt'].agg({'count1':np.sum})
all_data = pd.concat([actions1,test_data],axis=0)
all_data = all_data.reset_index(drop=True)

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
    d = {1:989,2:999,3:854,4:878,5:874,6:597,7:1425}
    if d.has_key(data):
        return d[data]
    else:
        return -1
all_data_1['day_std'] = all_data_1['day_of_week'].map(flag_to_number1)
train_df = all_data_1.iloc[:1032]
test_df = all_data_1.iloc[1032:,:]

def date_range(step,end):
    if end-step < 0:
        print 'date range not fit....'
    else:
        a = end-step
        b = end-1
        return a,b
def feature_1(date_end):
    frame_final = all_data_1.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7,]:
        begin,end = date_range(day,date_end)
        frame = all_data_1[(all_data_1.index >= begin) & (all_data_1.index<= end)]
        # 将取出的数据累加
        frame1 = frame['count1'].agg(
            {
                str(day) + 'day_mean': np.mean,
             })
        frame_final = pd.concat([frame_final, frame1], axis=0)
    return frame_final

frame_boss = feature_1(8)
for i in range(9,len(train_df)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
frame_boss = (frame_boss.T).reset_index(drop=True)

def mymodel(train_data):
    X_train = train_data[['date','day_of_week','n_time','day_std','1day_mean','2day_mean','3day_mean',
                        '4day_mean','7day_mean']].values
    y_train = train_data['count1'].values
    model1 = GradientBoostingRegressor().fit(X_train,y_train)
    return model1
my_model = mymodel(frame_boss)
test_df = test_df.drop(['count1'],axis=1)
y_lables = train_df.loc[:,'count1'].to_frame(name='count1')
y_pre = [1036,]
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
    # b = np.delete(a, [3,4, 5, 6, 9,10,11,12,13,14,15,16,17,18], axis=1)
    c = round(abs(my_model.predict(a)))
    y_pre.append(c)

res = pd.DataFrame({
    'date':test_all.loc[:,'date'],
    'res' : y_pre,
})
print res
res.to_csv('/root/word2vec/yancheng/data/pre/GBR2_5.csv',index=None,header=None,sep='\t')