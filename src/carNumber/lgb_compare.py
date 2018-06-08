# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:12:23 2018
对比原始特征，以下哪个模型较好
@author: dell-1
LGB:867125.460656,609.462835443
XGB:891010.924815,628.413774858
RF:996211.340255,694.672363636
"""

from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import ensemble
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd


data = pd.read_csv('original_train.csv')
test =  pd.read_csv('original_test.csv')

X_train = data.drop(['count1'],axis=1)
X_test = test.drop(['count1'],axis=1)

y_train = data['count1']
y_test = test['count1']

model = ensemble.RandomForestRegressor().fit(X_train,y_train)
y_pre = model.predict(X_test)

print(mean_squared_error(y_pre,y_test))
print(mean_absolute_error(y_pre,y_test))