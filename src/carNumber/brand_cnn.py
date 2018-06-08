# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:34:31 2018

@author: dell-1
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import cnn_method

train_brand1,test_brand1,brand1_pre = cnn_method.run('brand1.csv',
               'E:\\spyder3_code\\carNumber\\model0515\\original_test.csv')
 
brand_pre = pd.DataFrame({
                          'brand1_pre':brand1_pre.flatten()
                          })
    
train_brand2,test_brand2,brand2_pre = cnn_method.run('brand2.csv',
               'E:\\spyder3_code\\carNumber\\model0515\\original_test.csv') 

train_brand3,test_brand3,brand3_pre = cnn_method.run('brand3.csv',
               'E:\\spyder3_code\\carNumber\\model0515\\original_test.csv')

train_brand4,test_brand4,brand4_pre = cnn_method.run('brand4.csv',
               'E:\\spyder3_code\\carNumber\\model0515\\original_test.csv') 

train_brand5,test_brand5,brand5_pre = cnn_method.run('brand5.csv',
               'E:\\spyder3_code\\carNumber\\model0515\\original_test.csv')  

brand_pre['brand2_pre'] = brand2_pre.flatten()
brand_pre['brand3_pre'] = brand3_pre.flatten()
brand_pre['brand4_pre'] = brand4_pre.flatten()
brand_pre['brand5_pre'] = brand5_pre.flatten()
brand_pre['sum'] = brand_pre['brand1_pre']+brand_pre['brand2_pre']+brand_pre['brand3_pre']+brand_pre['brand4_pre']+brand_pre['brand5_pre']

test = pd.read_csv('E:\\spyder3_code\\carNumber\\model0515\\original_test.csv')

print(mean_squared_error(brand_pre['sum'],test['count1']))
               
