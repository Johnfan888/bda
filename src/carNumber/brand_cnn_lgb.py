# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:17:17 2018
928728.454721
@author: dell-1
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import cnn_method
from lightgbm.sklearn import LGBMRegressor

test = pd.read_csv('test.csv')

train_brand1,test_brand1,_ = cnn_method.run('brand1.csv',
               'test.csv')
 
    
train_brand2,test_brand2,_ = cnn_method.run('brand2.csv',
               'test.csv') 

train_brand3,test_brand3,_ = cnn_method.run('brand3.csv',
               'test.csv')

train_brand4,test_brand4,_ = cnn_method.run('brand4.csv',
               'test.csv') 

train_brand5,test_brand5,_ = cnn_method.run('brand5.csv',
               'test.csv')  

def get_model(brand_string,train_brand,test_brand):
    brand1 = pd.read_csv(brand_string)
    brand1 = brand1.iloc[90:,:].reset_index(drop=True)
    X_brand1 = brand1.drop(['brand','cnt'],axis=1)
    y_train = brand1['cnt'].values

    X_train = pd.concat([X_brand1,train_brand],axis=1)

    X_test = test.drop(['cnt'],axis=1)
    X_test = pd.concat([X_test,test_brand],axis=1)

   
    model = LGBMRegressor().fit(X_train,y_train)
    brand1_pre = model.predict(X_test)
    return brand1_pre
    

brand1_pre = get_model('brand1.csv',train_brand1,test_brand1)
brand2_pre = get_model('brand2.csv',train_brand2,test_brand2)
brand3_pre = get_model('brand3.csv',train_brand3,test_brand3)
brand4_pre = get_model('brand4.csv',train_brand4,test_brand4)
brand5_pre = get_model('brand5.csv',train_brand5,test_brand5)

brand_pre = pd.DataFrame({
                          'brand1_pre':brand1_pre.flatten()
                          })
    
brand_pre['brand2_pre'] = brand2_pre.flatten()
brand_pre['brand3_pre'] = brand3_pre.flatten()
brand_pre['brand4_pre'] = brand4_pre.flatten()
brand_pre['brand5_pre'] = brand5_pre.flatten()
brand_pre['sum'] = brand_pre['brand1_pre']+brand_pre['brand2_pre']+brand_pre['brand3_pre']+brand_pre['brand4_pre']+brand_pre['brand5_pre']

y_test = test['cnt']
print(mean_squared_error(brand_pre['sum'],y_test))
               
import matplotlib.pyplot as plt
plt.plot(range(0,len(y_test)),y_test)
plt.plot(range(0,len(y_test)),brand_pre['sum'])
plt.show() 
    
    