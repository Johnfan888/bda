# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#读取文件
user_table = pd.read_csv(r'E:\python\recommend\data\ratings.txt',names=['user_id','item_id','rating'],sep=' ')
user_table = user_table.groupby(['user_id','item_id'],as_index=False)['rating'].mean()

#切分数据
from sklearn.model_selection import train_test_split
user_train,user_test = train_test_split(user_table,test_size=0.2,random_state=4)
finall_test=user_test
user_test = user_test.drop(['rating'],axis=1)

#将数据转化为矩阵
data = pd.concat([user_train,user_test])
user_matrix = data.pivot(index='user_id', columns='item_id', values='rating')

# 构建共同的评分向量 
def build_xy(user_id1, user_id2): 
    bool_array = user_matrix.loc[user_id1].notnull() & user_matrix.loc[user_id2].notnull() 
    return user_matrix.loc[user_id1, bool_array], user_matrix.loc[user_id2, bool_array]

#相对信任度
train2 = pd.read_csv(r'E:\python\recommend\data\trust.txt',names=['trustor','trustee','weight'],sep=' ')

O = pd.Series(0.0,index=list(set((user_train)['user_id'].values)))
 
Mmax = train2['trustee'].value_counts().max()
Mu = train2.groupby('trustee',as_index=False)['trustor'].count().rename(columns={'trustor': 'trustor_count'})
Mu['R(u)'] = Mu.trustor_count/Mmax
   
Mu1 = Mu.groupby('trustee')['R(u)'].sum()

for i in Mu1.index:
    O[i] = 1-Mu1[i]

#相对活跃度
#相对专业度
id1 = {}
id2 = {}
l = set(user_train['user_id'].values)
for i in l:
    id1[i] = {}
    id2[i] = {}
    for j in l:
        if i == j:
            continue
        a,b = build_xy(i,j)
        id1[i][j] = len(a)
        if len(a) == 0:
            id2[i][j] = 0
        else:
            c = abs((a-b).values).sum()
            id2[i][j] = c / len(a)
          
#信任公式    
a={}
b={}
c={}
x,y,z=[0.3,0.4,0.3]
for i in id1.keys():
    a[i] = {}
    b[i] = {}
    c[i] = {}
    for j in id1[i].keys():
        a[i][j] = id1[i][j]/(max(id1[i]))
        b[i][j] = id2[i][j]/(max(id2[i]))
        c[i][j] = x*a[i][j] + y*b[i][j] + z*O[j]
        
#取前N个专家
exp_dict = {}

for i in set(user_train['user_id']): 
    c1 = pd.Series(c[i])            
    c1 = c1.sort_values(ascending = False)
    c2 = c1[:100]  
    exp_dict[i] = list(c2.index)
    
#计算相似度
def cosine(user_id1, user_id2): 
    x, y = build_xy(user_id1, user_id2) 
    # 分母 
    denominator = ((sum(x*x))*(sum(y*y)))**0.5
    try: 
        value = sum(x*y)/denominator 
    except ZeroDivisionError:
        value = 0 
    return value,len(x)

X_test1 = user_train.groupby('user_id')['item_id'].count().rename(columns={'item_id':'count'})

# 测试集与前N个专家的余弦相似度    
l1 = dict()
for i in set(user_test['user_id'].values):
    l1[i] = dict()
    if i in exp_dict.keys():
        e1 = exp_dict[i]
        e1 = pd.DataFrame(e1,columns=['user_id'])
        expert_A = e1.merge(user_train, on='user_id', how='left')
        expert_A1 = expert_A.groupby('user_id')['item_id'].count().rename(columns={'item_id':'count'})
        for j in exp_dict[i]: 
            cos,simCount = cosine(i,j)
            l1[i][j] = cos
#            l1[i][j] = cos*(2*simCount/(X_test1[i]+expert_A1[j]))
#            l1[i][j] = cos*(2*simCount/(X_test1[i]+expert_A1[j]))*c[i][j]

# 分母专家余弦相似度和        
l1_A = {}
for i in l1.keys():
    l1_A[i] =  sum(l1[i].values())
  
# 用户打分均值
Ck = user_train.groupby('user_id')['rating'].mean().rename(columns={'rating':'rating_mean'})
    
# 用户对商品的打分    
L1_Ck={}    
for i in set(user_test['user_id']):
    L1_Ck[i]={}
for i,j in user_test.values: 
    if i in exp_dict.keys():
        e1 = exp_dict[i]
        e1 = pd.DataFrame(e1,columns=['user_id'])
        expert_A = e1.merge(user_train, on='user_id', how='left')
		# 专家平均分    
        expertA_avg = expert_A.groupby('user_id',as_index=False)['rating'].mean().rename(columns={'rating':'rating_mean'})     
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
    else:
        e2 = user_train[user_train['item_id'] == j]        
        L1_Ck[i][j] = e2['rating'].mean()
    
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

























    
    
