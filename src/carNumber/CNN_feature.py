# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:24:22 2018

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
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error

def get_data(seq_len):   
   data = pd.read_csv('original_train.csv')
   test =  pd.read_csv('original_test.csv')
    #将周六周天的数据删除 
#   data = data[data['day_of_week']<7].reset_index(drop=True)
#   test = test[test['day_of_week']<7].reset_index(drop=True)

   data_1 = data['count1']
   test_1 = test['count1']
#   temp = data.loc[len(data_1)-seq_len:,'count1']
#   test_1 = pd.concat([temp,test['count1']]).reset_index(drop=True)      
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
    
seq_len1 = 90
data,test,train_original,test_original = get_data(seq_len1)

X_train,y_train = get_seq_data(data,seq_len1)
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train = X_train.reshape(-1,1,1,seq_len1)

input_shape = (1,1,seq_len1) 
def get_model():
#=======================model==============================  

    #=====model_left=======
    model_1 = Sequential()
    model_1.add(Conv2D(                
                 9,
                 kernel_size=(1,7),
                 strides=(1,1),
                padding = 'same',
                                activation = 'relu',
                input_shape = input_shape,
                 ))
    
    #=======model==============
    model_2 = Sequential()
    model_2.add(Conv2D(
                9,
                 kernel_size=(1,14),
                 strides=(1,1),
                padding = 'same',
                                activation = 'relu',
                input_shape = input_shape,
                 ))
    #========model_right=============
    model_3 = Sequential()
    model_3.add(Conv2D(
                9,
                 kernel_size=(1,30),
                 strides=(1,1),
                padding = 'same',
                                activation = 'relu',
                input_shape = input_shape,
                 ))
   
       
    

    #=======合并model==============
    model = Sequential()
    model.add(Merge([model_1,model_2,model_3,], 
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
    model.fit([X_train,X_train,X_train,], y_train, epochs=train_epoch,
              batch_size=batch_size,verbose=2,

              )
    
    dense1_layer_model = Model(inputs=model.input,  
                                    outputs=model.get_layer('a').output) 
    return model,dense1_layer_model

model,dense1_layer_model = get_model()


def date_range(step,end):
    if end-step < 0:
        print('date range not fit....')
    else:
        a = end-step
        b = end-1
        return a,b

y_lables = data.to_frame(name='count1')

test_size = len(test)
y_pre = []
nn_test = []

for j in range(0,test_size):
    length_y = len(y_lables)
    length_nn_test = len(nn_test)
    begin,end = date_range(seq_len1,length_y)
    frame = y_lables[(y_lables.index >= begin) & (y_lables.index<= end)]
    a = np.reshape(frame.as_matrix(), (1, -1))
    X_test = scaler.transform(a) #归一化 
    X_te = X_test.reshape(-1,1,1,seq_len1)    #reshape成输入形状
   
#    中间模型
    dense1_test = dense1_layer_model.predict([X_te,X_te,X_te])
    nn_test.append(dense1_test.flatten().tolist())
    
#CNN模型
    c = model.predict([X_te,X_te,X_te])[0]
    y_lables.ix[length_y] = pd.Series(c,index=y_lables.columns)
    y_pre.append(c.tolist())

    
dense1_train = dense1_layer_model.predict([X_train,X_train,X_train])
nn_train = pd.DataFrame(dense1_train, columns=['nn_%d' % column for column in range(27)])
#nn_train.to_csv('nn_train_7day.csv',index=None)

nn_test = pd.DataFrame(nn_test, columns=['nn_%d' % column for column in range(27)])
#nn_test.to_csv('nn_test_7day.csv',index=None) 



