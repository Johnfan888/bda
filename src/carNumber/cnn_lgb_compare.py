# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:56:19 2018
466226.04637
476.946587319

426283.152903
466.03749641
@author: dell-1
"""

from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import ensemble
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
from CNN_feature import nn_train,nn_test

datapath = 'E:\\spyder3_code\\carNumber\\data\\'
data = pd.read_csv(datapath+'train.csv')
test =  pd.read_csv(datapath+'test.csv')


data = data.iloc[90:,:].reset_index(drop=True)

predictor =  [x for x in data.columns if x not in
   ['count1','virtual_date',
   'djz','lunar_year','lunar_xun','year','after_hld'
    'l_qm_min','month','l_qm_max',
#    ,'qm_median','qm_max','l_qm_mean','before_hld','qm_mean',lunar_month
#     'qm_min'            
        ]]

X_train = data[predictor]
X_test = test[predictor]
#X_train = data[['date', 'day', 'lunar_day', 'day_of_week', 'qz_max' ,
#                'qz_mean', 'l_qz_min', 'l_qz_max', 'qz_median', 
#                'qz_min', 'l_qz_median', 'l_qz_mean',
#            ]]
#X_test = test[['date', 'day', 'lunar_day', 'day_of_week', 'qz_max' ,
#               'qz_mean', 'l_qz_min', 'l_qz_max', 'qz_median', 'qz_min',
#               'l_qz_median', 'l_qz_mean']]

#nn_train = pd.read_csv('nn_train_7day.csv')
#nn_test = pd.read_csv('nn_test_7day.csv')
#
#nn_train = nn_train[['nn_4', 'nn_8', 'nn_14', 'nn_7', 'nn_18', 'nn_16', 'nn_22', 'nn_15', 'nn_23']]
#nn_test = nn_test[['nn_4', 'nn_8', 'nn_14', 'nn_7', 'nn_18', 'nn_16', 'nn_22', 'nn_15', 'nn_23']]

X_train = pd.concat([X_train,nn_train],axis=1)
X_test = pd.concat([X_test,nn_test],axis=1)


y_train = data['count1']
y_test = test['count1']

other_params = {
                'n_estimators': 100,
                'learning_rate': 0.1, 
                'max_depth': 6, 
                'num_leaves': 31,
                'min_data_in_leaf':15,
                'Boosting' : 'gbdt'
}


model = LGBMRegressor(**other_params).fit(X_train,y_train)
y_pre = model.predict(X_test)

res = pd.DataFrame({
                    'pre':y_pre,
                    'y_test':y_test,
                    'day_of_week':test['day_of_week']
                    })

    
def flagtonumber(week,data):
    a = []
    for i in range(0,len(week)):
        if week[i]>5:
            a.append(data[i]*1.3)
        else:
            a.append(data[i]*0.9)
    return a
            
            
            
            
    
res['pre_chuli'] = flagtonumber(res['day_of_week'].values,res['pre'].values)



print(mean_squared_error(y_pre,y_test))
print(mean_absolute_error(y_pre,y_test))
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), 
                 X_train.columns), 
             reverse=True))
print('-----------------------------------------')
print(mean_squared_error(res['pre_chuli'],y_test))
print(mean_absolute_error(res['pre_chuli'],y_test))
import matplotlib.pyplot as plt
plt.plot(range(0,len(y_test)),y_test)
plt.plot(range(0,len(y_test)),res['pre_chuli'])
plt.show()
