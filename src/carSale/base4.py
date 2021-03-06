# -- coding:utf-8 --
import pandas as pd
from sklearn.model_selection import cross_val_score
import math
from sklearn import ensemble
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def get_train():
    train_df = pd.read_csv('/root/word2vec/carSale/data/train_2017.csv')
    id_sum = train_df['class_id'].values.tolist()
    id_sum = list(set(id_sum))
    frame_sum=train_df[(train_df.class_id==id_sum[0])].groupby(['sale_date']).sale_quantity.sum().round()
    for i in range(1,len(id_sum)):
        frame1=train_df[(train_df.class_id==id_sum[i])].groupby(['sale_date']).sale_quantity.sum().round()
        frame_sum = pd.concat([frame_sum,frame1],axis=1)
    frame_sum.columns = [id_sum]
    frame_sum = frame_sum.fillna(0).T.sort_index()
    frame_sum = frame_sum.reset_index()
    return frame_sum

frame_sum = get_train()

def get_rate(df_a,df_b):
    df_a = df_a.as_matrix()
    df_b = df_b.as_matrix()
    df = []
    for i in range(0,len(df_a)):
        if df_b[i] == 0:
            df.append(1)
        else:
            df.append((df_a[i]-df_b[i])/df_b[i])
    return df
# y_test = 5
def get_test_5():
    train1 = frame_sum.drop([201706,201707,201708,201709,201710],axis=1)
    train1['tb16_5'] = get_rate(train1[201605], train1[201505])
    train1['tb15_5'] = get_rate(train1[201505], train1[201405])
    train1['tb14_5'] = get_rate(train1[201405], train1[201305])
    train1['tb13_5'] = get_rate(train1[201305], train1[201205])

    train1['hb17_5'] = get_rate(train1[201704] , train1[201703])
    train1['hb17_4'] = get_rate(train1[201703] , train1[201702])
    train1['hb17_3'] = get_rate(train1[201702] , train1[201701])
    train1['hb17_2'] = get_rate(train1[201701] , train1[201612])
    # train1.columns = range(1, 72)
    # train1.columns = range(1, 67)
    train1.columns = range(1, 75)
    return train1
# y_test=6
def get_test_6():
    train2 = frame_sum.drop([201201,201707,201708,201709,201710],axis=1)
    train2['tb16_6'] = get_rate(train2[201606] , train2[201506])
    train2['tb15_6'] = get_rate(train2[201506] , train2[201406])
    train2['tb14_6'] = get_rate(train2[201406] , train2[201306])
    train2['tb13_6'] = get_rate(train2[201306] , train2[201206])

    train2['hb17_6'] = get_rate(train2[201705] , train2[201704])
    train2['hb17_5'] = get_rate(train2[201704] , train2[201703])
    train2['hb17_4'] = get_rate(train2[201703] , train2[201702])
    train2['hb17_3'] = get_rate(train2[201702] , train2[201701])

    # train2.columns= range(1,72)
    # train2.columns = range(1, 67)
    train2.columns = range(1, 75)
    return train2
# # y_test = 7
def get_test_7():
    train3 = frame_sum.drop([201201,201202,201708,201709,201710],axis=1)
    train3['tb16_7'] = get_rate(train3[201607] , train3[201507])
    train3['tb15_7'] = get_rate(train3[201507] , train3[201407])
    train3['tb14_7'] = get_rate(train3[201407] , train3[201307])
    train3['tb13_7'] = get_rate(train3[201307] , train3[201207])

    train3['hb17_7'] = train3[201706] - train3[201705]
    train3['hb17_6'] = train3[201705] - train3[201704]
    train3['hb17_5'] = train3[201704] - train3[201703]
    train3['hb17_4'] = train3[201703] - train3[201702]
    # train3.columns= range(1,72)
    train3.columns= range(1,75)
    # train3.columns = range(1, 67)
    return train3
# # y_test=8
def get_test_8():
    train4 = frame_sum.drop([201201,201202,201203,201709,201710],axis=1)
    train4['tb16_8'] = get_rate(train4[201608] , train4[201508])
    train4['tb15_8'] = get_rate(train4[201508] , train4[201408])
    train4['tb14_8'] = get_rate(train4[201408] , train4[201308])
    train4['tb13_8'] = get_rate(train4[201308] , train4[201208])

    train4['hb17_8'] = train4[201707] - train4[201706]
    train4['hb17_7'] = train4[201706] - train4[201705]
    train4['hb17_6'] = train4[201705] - train4[201704]
    train4['hb17_5'] = train4[201704] - train4[201703]
    # train4.columns= range(1,72)
    train4.columns= range(1,75)
    # train4.columns = range(1, 67)
    return train4
# # y_test=9
def get_test_9():
    train5 = frame_sum.drop([201201,201202,201203,201204,201710],axis=1)
    train5['tb16_9'] =get_rate(train5[201609] , train5[201509])
    train5['tb15_9'] =get_rate(train5[201509] , train5[201409])
    train5['tb14_9'] =get_rate(train5[201409] , train5[201309])
    train5['tb13_9'] =get_rate(train5[201309] , train5[201209])

    train5['hb17_9'] = train5[201708] - train5[201707]
    train5['hb17_8'] = train5[201707] - train5[201706]
    train5['hb17_7'] = train5[201706] - train5[201705]
    train5['hb17_6'] = train5[201705] - train5[201704]
    # train5.columns= range(1,72)
    train5.columns= range(1,75)
    # train5.columns = range(1, 67)
    return train5
# # y_test=10
def get_test_10():
    train6 = frame_sum.drop([201201,201202,201203,201204,201205],axis=1)
    train6['tb16_10'] = get_rate(train6[201610] , train6[201510])
    train6['tb15_10'] = get_rate(train6[201510] , train6[201410])
    train6['tb14_10'] = get_rate(train6[201410] , train6[201310])
    train6['tb13_10'] = get_rate(train6[201310] , train6[201210])

    train6['hb17_10'] = train6[201710] - train6[201709]
    train6['hb17_9'] = train6[201709] - train6[201708]
    train6['hb17_8'] = train6[201708] - train6[201707]
    train6['hb17_7'] = train6[201707] - train6[201706]
    train6.columns= range(1,75)
    # train6.columns = range(1, 67)
    return train6
#
train1 = get_test_5()
# print train1.head()
train2 = get_test_6()
train3 = get_test_7()
train4 = get_test_8()
train5 = get_test_9()
train6 = get_test_10()
# 预测8月
train_data1 = pd.concat([train1,train2,train3],axis=0)
test_data1 = train4
X_train1 = train_data1.drop([66,],axis=1)
y_train1 = train_data1.loc[:,66]
X_test1 = test_data1.drop([66,],axis=1)
y_test1 = test_data1.loc[:,66]


# 预测9月
train_data2 = pd.concat([train2,train3,train4],axis=0)
test_data2 = train5
X_train2 = train_data2.drop([66,],axis=1)
y_train2 = train_data2.loc[:,66]
X_test2= test_data2.drop([66],axis=1)
y_test2 = test_data2.loc[:,66]


# # 预测10月
train_data3 = pd.concat([train3,train4,train5],axis=0)
test_data3 = train6
X_train3 = train_data3.drop([66,],axis=1)
y_train3 = train_data3.loc[:,66]
X_test3 = test_data3.drop([66,],axis=1)
y_test3 = test_data3.loc[:,66]

def model(X_train,y_train,X_test,y_test):
    clf1 = ensemble.RandomForestRegressor(
        n_estimators=1000,
        max_depth=7,
        max_features=0.2,
        max_leaf_nodes=100
    ).fit(X_train, y_train)
    # clf1 = ensemble.GradientBoostingRegressor().fit(X_train,y_train)
    # clf1 = lgb.LGBMRegressor().fit(X_train,y_train)
    y_pre = clf1.predict(X_test)
    mse1 = math.sqrt(mean_squared_error(y_pre, y_test))
    return mse1

mse1 = model(X_train1,y_train1,X_test1,y_test1)
mse2 = model(X_train2,y_train2,X_test2,y_test2)
mse3 = model(X_train3,y_train3,X_test3,y_test3)
print mse1,mse2,mse3
print (mse3+mse2+mse1)/3
