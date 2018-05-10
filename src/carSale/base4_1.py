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
    # 统计每个品牌，每月的数量
    train_count = train_df[['sale_date','class_id','sale_quantity']]
    frame_count = train_count.groupby(['sale_date', 'class_id', ]).count().reset_index()
    frame_count.rename(columns={'sale_quantity': 'counts'}, inplace=True)
    # 统计8，9，10月的总数
    sum_8 = train_df[(train_df.sale_date==201708)].groupby(['class_id']).sale_quantity.sum().round().to_frame(name='a8')
    sum_9 = train_df[(train_df.sale_date==201709)].groupby(['class_id']).sale_quantity.sum().round().to_frame(name='a9')
    sum_10 = train_df[(train_df.sale_date==201710)].groupby(['class_id']).sale_quantity.sum().round().to_frame(name='a10')
    sum_sum = pd.concat([sum_8,sum_9,sum_10],axis=1).reset_index()
    # 统计平均数目，每个品牌，每个月的
    id_sum = train_df['class_id'].values.tolist()
    id_sum = list(set(id_sum))
    frame_sum=train_df[(train_df.class_id==id_sum[0])].groupby(['sale_date']).sale_quantity.mean().round()
    for i in range(1,len(id_sum)):
        frame1=train_df[(train_df.class_id==id_sum[i])].groupby(['sale_date']).sale_quantity.mean().round()
        frame_sum = pd.concat([frame_sum,frame1],axis=1)

    frame_sum.columns = [id_sum]
    frame_sum = frame_sum.fillna(0).T.sort_index()

    frame_sum = frame_sum.reset_index()
    return frame_sum,frame_count,sum_sum

frame_sum ,frame_count,sum_sum = get_train()

# y_test = 5
def get_test_5():
    train1 = frame_sum.drop([201706,201707,201708,201709,201710],axis=1)
    train1.columns = range(1, 67)
    # train1.columns = range(1, 75)
    return train1

# y_test=6
def get_test_6():
    train2 = frame_sum.drop([201201,201707,201708,201709,201710],axis=1)
    # train2['tb16_6'] = get_rate(train2[201606] , train2[201506])
    # train2['tb15_6'] = get_rate(train2[201506] , train2[201406])
    # train2['tb14_6'] = get_rate(train2[201406] , train2[201306])
    # train2['tb13_6'] = get_rate(train2[201306] , train2[201206])
    #
    # train2['hb17_6'] = get_rate(train2[201705] , train2[201704])
    # train2['hb17_5'] = get_rate(train2[201704] , train2[201703])
    # train2['hb17_4'] = get_rate(train2[201703] , train2[201702])
    # train2['hb17_3'] = get_rate(train2[201702] , train2[201701])

    # train2.columns= range(1,72)
    train2.columns = range(1, 67)
    # train2.columns = range(1, 75)
    return train2
# # y_test = 7
def get_test_7():
    train3 = frame_sum.drop([201201,201202,201708,201709,201710],axis=1)
    # train3['tb16_7'] = get_rate(train3[201607] , train3[201507])
    # train3['tb15_7'] = get_rate(train3[201507] , train3[201407])
    # train3['tb14_7'] = get_rate(train3[201407] , train3[201307])
    # train3['tb13_7'] = get_rate(train3[201307] , train3[201207])
    #
    # train3['hb17_7'] = train3[201706] - train3[201705]
    # train3['hb17_6'] = train3[201705] - train3[201704]
    # train3['hb17_5'] = train3[201704] - train3[201703]
    # train3['hb17_4'] = train3[201703] - train3[201702]
    # train3.columns= range(1,72)
    # train3.columns= range(1,75)
    train3.columns = range(1, 67)
    return train3
# # y_test=8
def get_test_8():
    train4 = frame_sum.drop([201201,201202,201203,201709,201710],axis=1)
    # train4['tb16_8'] = get_rate(train4[201608] , train4[201508])
    # train4['tb15_8'] = get_rate(train4[201508] , train4[201408])
    # train4['tb14_8'] = get_rate(train4[201408] , train4[201308])
    # train4['tb13_8'] = get_rate(train4[201308] , train4[201208])
    #
    # train4['hb17_8'] = train4[201707] - train4[201706]
    # train4['hb17_7'] = train4[201706] - train4[201705]
    # train4['hb17_6'] = train4[201705] - train4[201704]
    # train4['hb17_5'] = train4[201704] - train4[201703]
    # train4.columns= range(1,72)
    # train4.columns= range(1,75)
    train4.columns = range(1, 67)
    return train4
# y_test=9
def get_test_9():
    train5 = frame_sum.drop([201201,201202,201203,201204,201710],axis=1)
    # train5['tb16_9'] =get_rate(train5[201609] , train5[201509])
    # train5['tb15_9'] =get_rate(train5[201509] , train5[201409])
    # train5['tb14_9'] =get_rate(train5[201409] , train5[201309])
    # train5['tb13_9'] =get_rate(train5[201309] , train5[201209])
    #
    # train5['hb17_9'] = train5[201708] - train5[201707]
    # train5['hb17_8'] = train5[201707] - train5[201706]
    # train5['hb17_7'] = train5[201706] - train5[201705]
    # train5['hb17_6'] = train5[201705] - train5[201704]
    # train5.columns= range(1,72)
    # train5.columns= range(1,75)
    train5.columns = range(1, 67)
    return train5
# # y_test=10
def get_test_10():
    train6 = frame_sum.drop([201201,201202,201203,201204,201205],axis=1)
    # train6['tb16_10'] = get_rate(train6[201610] , train6[201510])
    # train6['tb15_10'] = get_rate(train6[201510] , train6[201410])
    # train6['tb14_10'] = get_rate(train6[201410] , train6[201310])
    # train6['tb13_10'] = get_rate(train6[201310] , train6[201210])
    #
    # train6['hb17_10'] = train6[201710] - train6[201709]
    # train6['hb17_9'] = train6[201709] - train6[201708]
    # train6['hb17_8'] = train6[201708] - train6[201707]
    # train6['hb17_7'] = train6[201707] - train6[201706]
    # train6.columns= range(1,75)
    train6.columns = range(1, 67)
    return train6
#
train1 = get_test_5()
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

def model(X_train,y_train,X_test,y_test,date,str_month):
    clf1 = ensemble.RandomForestRegressor(
    n_estimators=1000,
    max_depth=7,
    max_features=0.2,
    max_leaf_nodes=100
    ).fit(X_train, y_train)
    y_pre1 = clf1.predict(X_test)
    # 得到均值
    res = pd.DataFrame({
    'class_id':X_test1.loc[:,1],
    'y_pre':y_pre1
    })
    #得到count数目
    mean_8 = frame_count[frame_count['sale_date'] == date]
    mean_8 = mean_8.drop(['sale_date'],axis=1)
    a = sum_sum.loc[:,str_month]
    res = pd.merge(res,mean_8,on=['class_id'])
    res['y_mul'] = res['y_pre']*res['counts']
    res['y_test'] = a
    res = res.dropna(axis=0,how='any')
    mse = math.sqrt(mean_squared_error(res['y_mul'],res['y_test']))
    return mse



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

mse1 = model(X_train1,y_train1,X_test1,y_test1,201708,'a8')
mse2 = model(X_train2,y_train2,X_test2,y_test2,201709,'a9')
mse3 = model(X_train3,y_train3,X_test3,y_test3,201710,'a10')
print mse1,mse2,mse3
print (mse3+mse2+mse1)/3
print (mse3+mse2)/3
