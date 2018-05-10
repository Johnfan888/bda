# -- coding:utf-8 --
import pandas as pd
from sklearn.model_selection import cross_val_score
import math
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
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
# y_test = 7
train1 = frame_sum.drop(['level_0',201708,201709,201710],axis=1)
train1.columns= range(1,68)
train1['id'] = id
# y_test=8
train2 = frame_sum.drop(['level_0',201201,201709,201710],axis=1)
train2.columns= range(1,68)
train2['id'] = id
# y_test=9
train3 = frame_sum.drop(['level_0',201201,201202,201710],axis=1)
train3.columns= range(1,68)
train3['id'] = id
# y_test=10
train4 = frame_sum.drop(['level_0',201201,201202,201203,],axis=1)
train4.columns= range(1,68)
train4['id'] = id

# 预测10月
train_data1 = pd.concat([train1,train2,train3],axis=0)
test_data1 = train4
# print train_data1
X_train1 = train_data1.drop(['id',67],axis=1)
# X_train1 = train_data1[[56,57,58,59,60,61,62,63,64,65,66]]
y_train1 = train_data1.iloc[:,-2]
X_test1 = test_data1.drop(['id',67],axis=1)
# X_test1 = test_data1[[56,57,58,59,60,61,62,63,64,65,66]]
y_test1 = test_data1.iloc[:,-2]


# 预测9月
train_data2 = pd.concat([train1,train2,train4],axis=0)
test_data2 = train3
# X_train2 = train_data2[[56,57,58,59,60,61,62,63,64,65,66]]
X_train2 = train_data2.drop(['id',67],axis=1)
y_train2 = train_data2.iloc[:,-2]
X_test2= test_data2.drop(['id',67],axis=1)
# X_test2 = test_data2[[56,57,58,59,60,61,62,63,64,65,66]]
y_test2 = test_data2.iloc[:,-2]


# 预测8月
train_data3 = pd.concat([train1,train3,train4],axis=0)
test_data3 = train2
# X_train3 = train_data3[[56,57,58,59,60,61,62,63,64,65,66]]
X_train3 = train_data3.drop(['id',67],axis=1)
y_train3 = train_data3.iloc[:,-2]
X_test3 = test_data3.drop(['id',67],axis=1)
# X_test3 = test_data3[[56,57,58,59,60,61,62,63,64,65,66]]
y_test3 = test_data3.iloc[:,-2]


# 预测7月
train_data4 = pd.concat([train2,train3,train4],axis=0)
test_data4 = train1
# X_train4 = train_data4[[56,57,58,59,60,61,62,63,64,65,66]]
X_train4 = train_data4.drop(['id',67],axis=1)
y_train4 = train_data4.iloc[:,-2]
X_test4 = test_data4.drop(['id',67],axis=1)
# X_test4 = test_data4[[56,57,58,59,60,61,62,63,64,65,66]]
y_test4 = test_data4.iloc[:,-2]

def model(X_train,y_train,X_test,y_test):
    clf1 = ensemble.RandomForestRegressor(
        n_estimators=1000,
        max_depth=7,
        max_features=0.2,
        max_leaf_nodes=100
    ).fit(X_train, y_train)
    y_pre = clf1.predict(X_test)
    mse1 = abs(mean_squared_error(y_pre, y_test))
    return mse1
mse1 = model(X_train1,y_train1,X_test1,y_test1)
mse2 = model(X_train2,y_train2,X_test2,y_test2)
mse3 = model(X_train3,y_train3,X_test3,y_test3)
mse4 = model(X_train4,y_train4,X_test4,y_test4)
print mse1,mse2,mse3,mse4
mse = math.sqrt((mse1+mse2+mse3)/3)
print mse