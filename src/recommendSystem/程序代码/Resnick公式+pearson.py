# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

#读取数据
train = pd.read_csv(r'E:\python\recommend\data\ratings.txt',names=['user_id','item_id','rating'],sep=' ')
train = train.groupby(['user_id','item_id'],as_index=False)['rating'].mean()

#切分数据
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train,test_size=0.2,random_state=4)
test = X_test
X_test = X_test.drop(['rating'],axis=1)

X = pd.concat([X_train,X_test])
a1 = X.pivot(index='user_id', columns='item_id', values='rating')

# 用户打分均值
Ck = X_train.groupby('user_id')['rating'].mean().rename(columns={'rating':'rating_mean'})

# 构建共同的评分向量 
def build_xy(user_id1, user_id2): 
    bool_array = a1.loc[user_id1].notnull() & a1.loc[user_id2].notnull() 
    return a1.loc[user_id1, bool_array], a1.loc[user_id2, bool_array]

#计算特征和类的平均值
def calcMean(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
#计算Pearson系数
def calcPearson(user_id1, user_id2):
    x, y = build_xy(user_id1, user_id2)
    p=0
    if(len(x)!=0):
        x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
        sumTop = 0.0
        sumBottom = 0.0
        x_pow = 0.0
        y_pow = 0.0
        for i in x.index:
            sumTop += (x[i]-x_mean)*(y[i]-y_mean)
        for i in x.index:
            x_pow += math.pow(x[i]-x_mean,2)
        for i in x.index:
            y_pow += math.pow(y[i]-y_mean,2)
        sumBottom = math.sqrt(x_pow*y_pow)
        if sumBottom == 0:
            p = 0
        else:
            p = sumTop/sumBottom
    return p    
    
# 测试集与用户的Pearson相似度    
l1 = dict()
for i in set(X_test['user_id'].values):
    l1[i] = dict()
    for j in set(X_train['user_id'].values): 
        if calcPearson(i,j)>0:
            l1[i][j] = calcPearson(i,j)
        else:
            l1[i][j]=0

# 分母专家Pearson相似度和        
l1_A = {}
for i in l1.keys():
    l1_A[i] = sum(l1[i].values())

# 用户对商品的打分    
L1_Ck={}    
for i in set(X_test['user_id']):
    L1_Ck[i]={}
for i,j in X_test.values: 
    e1 = X_train[X_train['item_id']==j][['user_id','rating']]
    sum=0
    for x,y in e1.values:
        if l1_A[i] != 0:
            sum += (y-Ck[i])* l1[i][x]
    if l1_A[i] != 0:
        if i in Ck.index:
            sum = sum/l1_A[i] + Ck[i]
        else:
            sum = sum/l1_A[i]
    L1_Ck[i][j]=sum
   


#预测评分
df = pd.DataFrame(L1_Ck)
df = df.unstack().dropna()
df = df.reset_index()
df.columns = ['user_id','item_id','rating']
df = df.merge(finall_test, on=['user_id','item_id'],how='left')
df1 = (df['rating_x']-df['rating_y'])**2
df2 = abs(df['rating_x']-df['rating_y'])

#RMSE
s1=0
for i in df1:
    s1+=i
import math
s1 = math.sqrt(s1/7099)

#MAE
df2 = abs(df['rating_x']-df['rating_y'])
s2=0
for i in df2:
    s2+=i
s2 = math.sqrt(s2/7099) 





