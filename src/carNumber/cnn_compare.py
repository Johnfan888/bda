# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:48:51 2018
对比CNN提取的特征，那个较好
RF:
1744440.09469
1031.89163636
XGB:
1677676.36245
1012.79782315
LGB:
1663242.98856
994.508876512
@author: dell-1
"""

from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import ensemble
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd

data = pd.read_csv('original_train.csv')
test =  pd.read_csv('original_test.csv')

nn_train = pd.read_csv('nn_train_7day.csv')
nn_test = pd.read_csv('nn_test_7day.csv')

nn_train = nn_train[['nn_4', 'nn_8', 'nn_14', 'nn_7', 'nn_18', 'nn_16', 'nn_22', 'nn_15', '']]

y_train = data.loc[90:,'count1'].values
y_test = test['count1']

model = LGBMRegressor().fit(nn_train,y_train)
y_pre = model.predict(nn_test)

print(mean_squared_error(y_pre,y_test))
print(mean_absolute_error(y_pre,y_test))
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), 
                 nn_train.columns), 
             reverse=True))



