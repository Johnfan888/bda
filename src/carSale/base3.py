# -- coding:utf-8 --
import pandas as pd
from sklearn.model_selection import cross_val_score
import math
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
def get_train():
    train_df = pd.read_csv('/root/word2vec/carSale/data/train_2017.csv')
    id_sum = train_df['class_id'].values.tolist()
    id_sum = list(set(id_sum))
    frame_sum=train_df[(train_df.class_id==id_sum[0])].groupby(['sale_date']).sale_quantity.sum().round()
    for i in range(1,len(id_sum)):
        frame1=train_df[(train_df.class_id==id_sum[i])].groupby(['sale_date']).sale_quantity.sum().round()
        frame_sum = pd.concat([frame_sum,frame1],axis=1)
    frame_sum.columns = [id_sum]
    frame_sum = frame_sum.fillna(0).T.sort_index()
    frame_sum = frame_sum.reset_index()
    return frame_sum
frame_sum = get_train()
# print frame_sum

id = frame_sum['level_0']
# y_test = 5
train1 = frame_sum.drop(['level_0',201706,201707,201708,201709,201710],axis=1)
train1.columns= range(1,66)
train1['id'] = id
# y_test=6
train2 = frame_sum.drop(['level_0',201201,201707,201708,201709,201710],axis=1)
train2.columns= range(1,66)
train2['id'] = id
# y_test = 7
train3 = frame_sum.drop(['level_0',201201,201202,201708,201709,201710],axis=1)
train3.columns= range(1,66)
train3['id'] = id
# y_test=8
train4 = frame_sum.drop(['level_0',201201,201202,201203,201709,201710],axis=1)
train4.columns= range(1,66)
train4['id'] = id
# y_test=9
train5 = frame_sum.drop(['level_0',201201,201202,201203,201204,201710],axis=1)
train5.columns= range(1,66)
train5['id'] = id
# y_test=10
train6 = frame_sum.drop(['level_0',201201,201202,201203,201204,201205],axis=1)
train6.columns= range(1,66)
train6['id'] = id






# 预测8月
train_data1 = pd.concat([train1,train2,train3],axis=0)
test_data1 = train4
# print train_data1
# X_train1 = train_data1.drop(['id',65],axis=1)
# X_train1 = train_data1[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
X_train1 = train_data1[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_train1 = train_data1.iloc[:,-2]
# X_test1 = test_data1.drop(['id',65],axis=1)
# X_test1 = test_data1[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
X_test1 = test_data1[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_test1 = test_data1.iloc[:,-2]
print '----------8-----------'
print mean_squared_error(y_test1,test_data1.iloc[:,-3])

# 预测9月
train_data2 = pd.concat([train2,train3,train4],axis=0)
test_data2 = train5
# X_train2 = train_data2[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
# X_train2 = train_data2.drop(['id',65],axis=1)
X_train2 = train_data2[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_train2 = train_data2.iloc[:,-2]
# X_test2= test_data2.drop(['id',65],axis=1)
# X_test2 = test_data2[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
X_test2 = test_data2[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_test2 = test_data2.iloc[:,-2]
print '----------9-----------'
print mean_squared_error(y_test2,test_data2.iloc[:,-3])

# 预测10月
train_data3 = pd.concat([train3,train4,train5],axis=0)
test_data3 = train6
# X_train3 = train_data3[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
# X_train3 = train_data3.drop(['id',65],axis=1)
X_train3 = train_data3[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_train3 = train_data3.iloc[:,-2]
# X_test3 = test_data3.drop(['id',65],axis=1)
# X_test3 = test_data3[[51,52,53,54,55,56,57,58,59,60,61,62,63,64]]
X_test3 = test_data3[[4, 5, 6, 8, 11, 15, 19, 23, 28, 29, 30, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                       46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                       58, 59, 60, 61, 62, 63, 64]]
y_test3 = test_data3.iloc[:,-2]
print '----------10-----------'
print mean_squared_error(y_test3,test_data3.iloc[:,-3])

def get_params():
    for i in np.arange(1,20,1):
        for j in np.arange(0.1,1,0.05):
            for k in np.arange(10,200,10):
                return i,j,k

def model(X_train,y_train,X_test,y_test,):
    a = 5
    b = 0.7
    c = 170
    clf1 = ensemble.RandomForestRegressor(
        n_estimators=1000,
        max_depth=a,
        max_features=b,
        max_leaf_nodes=c
    ).fit(X_train, y_train)
    # clf1 = ensemble.GradientBoostingRegressor().fit(X_train,y_train)
    y_pre = clf1.predict(X_test)
    names = X_train.columns
    # print "Features sorted by their score:"
    # a =  sorted(zip(map(lambda x: round(x, 4), clf1.feature_importances_), names),
    #              reverse=True)
    # feature_import = []
    # for i in a:
    #     feature_import.append(i[1])
    # print feature_import
    mse1 = abs(mean_squared_error(y_pre, y_test))
    return mse1
a,b,c = get_params()
# out1 = []
# for i in np.arange(1, 20, 4):
#     for j in np.arange(0.1, 1, 0.2):
#         for k in np.arange(10, 200, 40):
mse1 = model(X_train1,y_train1,X_test1,y_test1,)
mse2 = model(X_train2,y_train2,X_test2,y_test2,)
mse3 = model(X_train3,y_train3,X_test3,y_test3,)
mse4 = math.sqrt((mse1+mse2 + mse3) / 3)
mse5 = math.sqrt((mse2 + mse3) / 2)
print mse1,mse2,mse3
print mse4,mse5
# out1.append([i,j,k,mse4,mse5])
# print '---clf-----'
# y_tuples1 = sorted(out1, key=lambda x: x[3])
# y_tuples2 = sorted(out1, key=lambda x: x[4])
# res = pd.DataFrame({
#     'x3':y_tuples1,
#     'x4':y_tuples2
# })
# print y_tuples1
# print y_tuples2
# res.to_csv('/root/word2vec/carSale/data/mse4.csv')
# y_tuples2.to_csv('/root/word2vec/carSale/data/mse5.csv')