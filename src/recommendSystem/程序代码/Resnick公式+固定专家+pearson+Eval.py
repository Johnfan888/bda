# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

train = pd.read_csv(r'E:\python\recommend\data\ratings.txt',names=['user_id','item_id','rating'],sep=' ')
train = train.groupby(['user_id','item_id'],as_index=False)['rating'].mean()


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train,test_size=0.2,random_state=4)
test = X_test
X_test = X_test.drop(['rating'],axis=1)


# 活跃度
Nmax = X_train['user_id'].value_counts().max()
Nu = X_train.groupby('user_id',as_index=False)['item_id'].count().rename(columns={'item_id': 'user_count'})
Nu['F(u)'] = Nu.user_count/Nmax
           
# 专业度   
item_R = X_train.groupby('item_id',as_index=False)['rating'].agg({'user_count':'count','R_mean':'mean'})
Rmax = X_train['rating'].max()
train1 = X_train.merge(item_R, on='item_id', how='left')
train1['V(R)'] = 1 - abs(train1.rating - train1.R_mean)/Rmax
user_bias = train1.groupby('user_id', as_index=False)['V(R)'].mean().rename(columns={'V(R)':'E(u)'})


# 信誉度
train2 = pd.read_csv(r'E:\python\recommend\data\trust.txt',names=['trustor','trustee','weight'],sep=' ')
Mmax = train2['trustee'].value_counts().max()
Mu = train2.groupby('trustee',as_index=False)['trustor'].count().rename(columns={'trustor': 'trustor_count'})
Mu['R(u)'] = Mu.trustor_count/Mmax

# 表的整合和处理   
data = pd.concat([Nu,user_bias['E(u)']],axis=1)
a = pd.DataFrame()
a['user_id'] = Mu['trustee']
a['R(u)'] = Mu['R(u)']  
data1 = a.merge(data, on='user_id', how='left')
del data1['user_count']
data1 = data1.dropna(axis=0,how='any')


# 计算专家信任值
data1['Eval(u)'] = 0.3*data1['F(u)'] + 0.4*data1['E(u)'] + 0.3*data1['R(u)']      
data1 = data1.sort_index(axis = 0,ascending = False,by = 'Eval(u)').reset_index(drop = True)      

# 专家信任值曲线图
data1['Eval(u)'].plot()

# 专家分级
data1['level'] = pd.cut(data1['Eval(u)'],[0.4,0.8,1],labels=['A','B'])

exp = data1[['user_id','Eval(u)']]
exp = exp.groupby('user_id')['Eval(u)'].mean()           
# 专家数据库

expert = pd.DataFrame()

expert = data1[['user_id','Eval(u)','level']]
expert = expert.merge(train, on='user_id', how='left')


expert_A = expert[expert['level'] == 'A']


X = pd.concat([X_train,X_test])
a1 = X.pivot(index='user_id', columns='item_id', values='rating')

expert_A1 = expert_A.pivot(index='user_id', columns='item_id', values='rating')



# 用户打分均值

Ck = X_train.groupby('user_id')['rating'].mean().rename(columns={'rating':'rating_mean'})


# 构建共同的评分向量 
def build_xy(user_id1, user_id2): 
    bool_array = a1.loc[user_id1].notnull() & a1.loc[user_id2].notnull() 
    return a1.loc[user_id1, bool_array], a1.loc[user_id2, bool_array]

## 余弦相似度 
#def cosine(user_id1, user_id2): 
#    x, y = build_xy(user_id1, user_id2) 
#    # 分母 
#    denominator = (sum(x*x)*sum(y*y))**0.5
#    try: 
#        value = sum(x*y)/denominator 
#    except ZeroDivisionError:
#        value = 0 
#    return value

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

# 测试集与专家的余弦相似度    
l1 = dict()
for i in set(X_test['user_id'].values):
    l1[i] = dict()
    for j in set(expert_A['user_id'].values): 
        if calcPearson(i,j)>0:
            l1[i][j] = calcPearson(i,j)+exp[j]
        else:
            l1[i][j]=0

# 分母专家余弦相似度和        
l1_A = {}
for i in l1.keys():
    l1_A[i] = sum(l1[i].values())
  
# 专家平均分    
expertA_avg = expert_A.groupby('user_id',as_index=False)['rating'].mean().rename(columns={'rating':'rating_mean'})     


# 用户对商品的打分    
L1_Ck={}    
for i in set(X_test['user_id']):
    L1_Ck[i]={}
for i,j in X_test.values: 
    e1 = expert_A[expert_A['item_id']==j][['user_id','rating']]
    e1 = e1.merge(expertA_avg,on='user_id',how='left')
    sum=0
    for x,y,z in e1.values:
        if l1_A[i] != 0:
            sum += (y-z)* l1[i][x]
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
