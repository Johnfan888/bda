# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:55:28 2018
模拟conv1d的样子
想法：实现CNN自动提取特征滑动：
数据并没有填充没有的日期
@author: dell-1
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.optimizers import Adam
from keras import losses
from keras.layers import Conv2D,MaxPool2D,BatchNormalization
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor

def get_data():
    data_path = 'E:\\spyder3_code\\carNumber\\data\\'

    data = pd.read_csv(data_path+'train.csv',)
    test = pd.read_csv(data_path+'test.csv',)

    #将周六周天的数据删除 
    data = data[data['day_of_week']<7].reset_index(drop=True)
    test = test[test['day_of_week']<7].reset_index(drop=True)
    
    data_1 = data['count1']
    test_1 = test['count1']
    
    return data_1,test_1,data,test
    
def get_seq_data(train_data,seq_length):
#    获得序列数据
    dataX = []
    dataY = []
    for i in range(0, len(train_data) - seq_length, 1):
        seq_in = train_data[i:i + seq_length]
        seq_out = train_data[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return np.array(dataX),np.array(dataY)
    
data,test,train_original,test_original = get_data()
   

seq_len = 30
X_train,y_train = get_seq_data(data,seq_len)
X_test,y_test = get_seq_data(test,seq_len)
print('train的样本数：'+ str(X_train.shape))
print('test的样本数：'+ str(X_test.shape))

scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_tr = X_train.reshape(-1,1,1,30)
X_te = X_test.reshape(-1,1,1,30)

y_train = y_train.reshape(-1,1).flatten() 
y_test = y_test.reshape(-1,1).flatten() 
input_shape = (1,1,30)

def get_model():
    ###########################CNN模型#############################
    model = Sequential()
    conv1_filter = 32
    conv1_k_size = (1,14)
    conv1_stride = (1,1)
    #第一层卷积
    model.add(Conv2D(
                 conv1_filter,
                 kernel_size=conv1_k_size,
                 strides=conv1_stride,
                padding = 'same',
#                activation = 'relu',
                input_shape = input_shape,
                 ))

    conv2_filter = 64
    conv2_k_size = (1,7)
    conv2_stride = (1,1)
    #第二层卷积
    model.add(Conv2D(
                conv2_filter,
                 kernel_size=conv2_k_size,
                 strides = conv2_stride,
                padding = 'same',
                activation = 'relu',
                name='conv2'
                 
                 ))
    model.add(BatchNormalization())

    model.add(MaxPool2D(
                        pool_size = (1,3),
                        padding = 'same',
                        name = 'pool1'
    ))

    model.add(BatchNormalization())

    drop_value = 0.25
    model.add(Dropout(drop_value))

        #添加一层全链接层
    model.add(Flatten())
    model.add(Dense(10,name = 'fc1'))
    model.add(Activation('relu'))

    model.add(Dense(1))
    #model.add(Activation('relu'))

    learning =0.001
    lr_decay=0.0001
    model.compile(optimizer=Adam(lr=learning, 
                             decay=lr_decay
                             ),
              loss=losses.mean_squared_error,)

    print('\nTraining-----------')
    train_epoch =500
    batch_size = 32
    model.fit(X_tr, y_train, epochs=train_epoch,
          batch_size=batch_size,verbose=2,validation_data=(X_te,y_test))
    
    dense1_layer_model = Model(inputs=model.input,  
                                    outputs=model.get_layer('fc1').output) 

    model.summary()
    return model,dense1_layer_model

#CNN 模型
model,dense1_layer_model = get_model()
# cnn提取特征+lgb
dense1_train = dense1_layer_model.predict(X_tr)
nn_train = pd.DataFrame(dense1_train, columns=['nn_%d' % column for column in range(10)])

dense1_test = dense1_layer_model.predict(X_te)
nn_test = pd.DataFrame(dense1_test, columns=['nn_%d' % column for column in range(10)])

clf = XGBRegressor().fit(nn_train,y_train)
y_pre = clf.predict(nn_test)

print('CNN(特征)+lgb:')
print(mean_squared_error(y_pre,y_test))

#原始特征
#=================================================================
def get_original_feature(train,test,seq_len):
    predictor = [column for column in train.columns 
                 if column not in ['year','lunar_year',
                                   'lunar_xun','djz','date','count1','virtual_date']]
    train = train[predictor]
    test = test[predictor]
    train = train[seq_len:]
    test = test[seq_len:]
    train = train.fillna(-1)
    
    scaler = preprocessing.MinMaxScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test) 
    return train,test
    
train_feature,test_feature = get_original_feature(train_original,test_original,seq_len)
#========人工特征+lgb============
clf2 = XGBRegressor().fit(train_feature,y_train)
y_pre2 = clf2.predict(test_feature)
print('人工特征+lgb:')
print(mean_squared_error(y_pre2,y_test))


#============CNN特征+原始特征+lgb======================
train_f = np.column_stack((train_feature,dense1_train))
test_f = np.column_stack((test_feature,dense1_test))
clf3 = XGBRegressor().fit(train_f,y_train)
y_pre3 = clf3.predict(test_f)
print('特征(CNN+人工)+lgb:')
print(mean_squared_error(y_pre3,y_test))


    