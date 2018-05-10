# -*- coding: utf-8 -*-
import pandas as pd
from pandas import concat
from pandas import DataFrame
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_data = pd.read_csv('/root/word2vec/yancheng/data/train_data.txt', sep='\t')
train_data.columns = ['date', 'day_of_week', 'brand', 'cnt']
train_data = train_data.drop(['brand'],axis=1)
train_dispose0 = train_data[['date','day_of_week', 'cnt']].\
                groupby(['date','day_of_week']).agg('sum').reset_index()
week_diff = [1,]
for i in range(len(train_dispose0)-1):
    a = train_dispose0.loc[i,'day_of_week']
    b =train_dispose0.loc[i+1,'day_of_week']
    week_diff.append(b-a)
train_dispose0['week_diff'] = week_diff
test_dispose = train_dispose0[train_dispose0['date'] >=832].reset_index()
train_dispose = train_dispose0[train_dispose0['date'] <=832]
print train_dispose0



fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train_dispose0.index,train_dispose0.loc[:,'week_diff'])
plt.show()
plt.savefig('/root/word2vec/yancheng/data/pic/week_diff1.jpg')



# print len(train_dispose)
# print train_dispose
test_dispose = train_dispose0[train_dispose0['date'] >832]
# print len(test_dispose)
# print test_dispose