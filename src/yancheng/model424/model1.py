# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:23:55 2018
想法：实现CNN自动提取特征滑动：
将数列排成一排，用（1，7）或（1，14）等滤波器去过滤得到特征后拼接
特征===》lightGBM
@author: dell-1
"""
import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Activation,Merge
from keras.optimizers import Adam
from keras import losses
from keras.layers import Conv2D,MaxPool2D,BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils 
from sklearn import preprocessing
#from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error

def get_data(seq_len):   
    data_path = 'E:\\spyder3_code\\carNumber\\data\\'

    data = pd.read_csv(data_path+'train.csv',)
    test = pd.read_csv(data_path+'test.csv',)

    #将周六周天的数据删除 
        #    data = data[data['day_of_week']<7].reset_index(drop=True)
        #    test = test[test['day_of_week']<7].reset_index(drop=True)

    data_1 = data['count1']
    temp = data.loc[len(data_1)-seq_len:,'count1']
    test_1 = pd.concat([temp,test['count1']]).reset_index(drop=True)      
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
    
seq_len1 = 275
data,test,train_original,test_original = get_data(seq_len1)

X_train,y_train = get_seq_data(data,seq_len1)
X_test,y_test = get_seq_data(test,seq_len1)

scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(-1,1,1,seq_len1)
X_test = X_test.reshape(-1,1,1,seq_len1)

input_shape = (1,1,seq_len1) 
def get_model():
#=======================model==============================  

    #=====model_left=======
    model_1 = Sequential()
    model_1.add(Conv2D(
                 32,
                 kernel_size=(1,7),
                 strides=(1,1),
                padding = 'same',
                #                activation = 'relu',
                input_shape = input_shape,
                 ))
    #=======model==============
    model_2 = Sequential()
    model_2.add(Conv2D(
                32,
                 kernel_size=(1,14),
                 strides=(1,1),
                padding = 'same',
                #                activation = 'relu',
                input_shape = input_shape,
                 ))
    #========model_right=============
    model_3 = Sequential()
    model_3.add(Conv2D(
                 32,
                 kernel_size=(1,30),
                 strides=(1,1),
                padding = 'same',
                #                activation = 'relu',
                input_shape = input_shape,
                 ))


    #=======合并model==============
    model = Sequential()
    model.add(Merge([model_1,model_2,model_3], 
                mode='concat',concat_axis=1))  
    model.add(Flatten(name='a'))
    model.add(Dense(1))
    learning =0.01
    lr_decay=0.001
    model.compile(optimizer=Adam(lr=learning, 
                         decay=lr_decay
                         ),
          loss=losses.mean_squared_error,)

    print('\nTraining-----------')
    train_epoch =100
    batch_size = 32
    model.fit([X_train,X_train,X_train], y_train, epochs=train_epoch,
              batch_size=batch_size,verbose=2,
              validation_data=([X_test,X_test,X_test], y_test,)
              )
    
    dense1_layer_model = Model(inputs=model.input,  
                                    outputs=model.get_layer('a').output) 
    return model,dense1_layer_model

model,dense1_layer_model = get_model()
model.summary()
# cnn提取特征+lgb
dense1_train = dense1_layer_model.predict([X_train,X_train,X_train])
nn_train = pd.DataFrame(dense1_train, columns=['nn_%d' % column for column in range(96)])

dense1_test = dense1_layer_model.predict([X_test,X_test,X_test])
nn_test = pd.DataFrame(dense1_test, columns=['nn_%d' % column for column in range(96)])

clf = LGBMRegressor().fit(nn_train,y_train)
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
    train = train.loc[seq_len:]
#    test = test.loc[seq_len:]
    train = train.fillna(-1)
    
    scaler = preprocessing.MinMaxScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test) 
    return train,test
    
train_feature,test_feature = get_original_feature(train_original,test_original,seq_len1)
#========人工特征+lgb============
clf2 = LGBMRegressor().fit(train_feature,y_train)
y_pre2 = clf2.predict(test_feature)
print('人工特征+lgb:')
print(mean_squared_error(y_pre2,y_test))


#============CNN特征+原始特征+lgb======================

train_f = np.column_stack((train_feature,dense1_train))
test_f = np.column_stack((test_feature,dense1_test))
clf3 = LGBMRegressor().fit(train_f,y_train)
y_pre3 = clf3.predict(test_f)
print('特征(CNN+人工)+lgb:')
print(mean_squared_error(y_pre3,y_test))
#print(sorted(zip(clf3.feature_importances_, names), 
#             reverse=True))