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
week_diff = [1,]
for i in range(len(train_dispose0)-1):
    a = train_dispose0.loc[i,'day_of_week']
    b =train_dispose0.loc[i+1,'day_of_week']
    week_diff.append(b-a)
train_dispose0['week_diff'] = week_diff
train_dispose = train_dispose0[train_dispose0['date'] <=832]
test_dispose = train_dispose0[train_dispose0['date'] >=832].reset_index()
y_test = test_dispose['cnt']
#test_data是test的x值
test_data = test_dispose[['date','day_of_week',]]
def date_range(step,end):
    if end-step < 0:
        print 'date range not fit....'
    else:
        a = end-step
        b = end-1
        return a,b
def feature_1(date_end):
    frame_final = train_dispose.iloc[date_end,:]
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60, 90]:
#    for day in [1, 2, 3, 4,6,7,8,13, 14,19, 20, 21, 30,60, 90]:
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
    data = data.drop(['1day_std'],axis=1)
    data['day_mean'] = data['day_of_week'].map(flag_to_number)
    data_all = data.drop(['date', 'day_of_week','week_diff', 'cnt','day_mean'], axis=1)
    data_frame = pd.concat([data['date'],
                            data['day_of_week'],
                            # data['week_diff'],
                            data['day_mean'],
                           data_all, data['cnt']],axis=1)
    return data_frame
def model(train_data):
    X_train1 = train_data.drop(['date','cnt',],axis=1)
    print X_train1
    y_train1 = train_data.loc[:,'cnt']
    model1 = RandomForestRegressor(n_estimators=1000,
                                    max_depth=7,
                                    max_features=0.2,
                                    max_leaf_nodes=100).fit(X_train1,y_train1)
    # model1 = linear_model.LinearRegression().fit(X_train1,y_train1)
    return model1,X_train1
def mylgb(train_data):
    # create dataset for lightgbm
    X_train1 = train_data.drop(['date','cnt',], axis=1)
    y_train1 = train_data.loc[:, 'cnt']
    model = lgb.LGBMRegressor(
        # objective='regression',
        # num_leaves=64,
        # learning_rate=0.05,
        # n_estimators=10000
    ).fit(X_train1,y_train1,)
    return model,X_train1
frame_boss = feature_1(90)
for i in range(91,len(train_dispose)):
    frame_final = feature_1(i)
    frame_boss = pd.concat([frame_boss,frame_final],axis=1)
frame_boss =  (frame_boss.T).reset_index(drop=True)
data_frame = data_process(frame_boss)

print data_frame

mymodel3,X_train1 = mylgb(data_frame)
test_data['day_mean'] = test_data['day_of_week'].map(flag_to_number)

test_data1 = test_data.drop(['day_of_week',],axis=1)
# test_data1 = test_data

y_lables = train_dispose.loc[:,'cnt'].to_frame(name='cnt')
#先将第1032天的数据加上，不用预测
y_pre = ["1032\t1306",]
y_pre1 = [2051,]
for j in range( 1, len(test_data1)):
    length_y = len(y_lables)
    frame_final1 = test_data.loc[j]
    x_list = []
    for day in [1, 2, 3, 4, 7, 14, 21, 30, 60, 90]:
        begin,end = date_range(day,length_y)
        frame = y_lables[(y_lables.index >= begin) & (y_lables.index<= end)]
        frame1 = frame['cnt'].agg(
            { str(day) + 'day_avg': np.mean,
              str(day) + 'day_std': np.std,
              str(day) + 'day_min': np.min,
             })
        frame_final1 = pd.concat([frame_final1,frame1],axis=0)
    frame_final1 = frame_final1.drop(['1day_std'])
    date = int(frame_final1[0])
    frame_final2 = frame_final1.drop(['date'])
    print frame_final2
    a = np.reshape(frame_final2.as_matrix(),(1,-1))
    b = abs(round(mymodel3.predict(a)[0]))
    c = str(date)+'\t'+str(int(b))
    y_pre.append(c)
    y_pre1.append(b)
    y_lables.loc[length_y] = b

predict = pd.DataFrame({
    'lgb_pre':y_pre1
})
predict.to_csv('/root/word2vec/yancheng/data/pre/mylgb.csv',index=None)

sum = 0.0
for i in range(0,len(y_pre1)):
    e = (y_pre1[i]-y_test[i])**2
    sum+=e
print sum/len(y_pre1)



