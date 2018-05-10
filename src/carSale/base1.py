# -- coding:utf-8 --
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import math
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
# print frame_sum
d = frame_sum['level_0']
train1 = frame_sum.drop(['level_0',201709,201710],axis=1)
train1.columns= range(1,69)
train1['id'] = id
train2 = frame_sum.drop(['level_0',201201,201710],axis=1)
train2.columns= range(1,69)
train2['id'] = id
train3 = frame_sum.drop(['level_0',201201,201202],axis=1)
train3.columns= range(1,69)
train3['id'] = id
traindata = pd.concat([train1,train2,train3],axis=0)


# X_train = traindata.drop([68,'id'],axis=1)
# y_train = traindata.iloc[:,-2]
# y_pre1 = traindata.iloc[:,-3]+2
# mse2 = mean_squared_error(y_train, y_pre1)
# mse3 = math.sqrt(mse2)
# print mse3

#clf = linear_model.LinearRegression()
# gbdt = GradientBoostingRegressor()
# mse = cross_validation.cross_val_score(gbdt, X_train, y_train, scoring='neg_mean_squared_error').mean()
# mse1 = math.sqrt(-1*mse)
# print mse1

#frame_sum.to_csv('train1.csv',index = None)

# X_train = traindata[[56,57,58,59,60,61,62,63,64,65,66,67]]
X_train = traindata.drop([68,'id'],axis=1)
y_train = traindata.iloc[:,-2]
# y_pre1 = traindata.iloc[:,-3]
from sklearn import ensemble
#clf = linear_model.LinearRegression()
# clf = lgb.LGBMRegressor()
clf = ensemble.RandomForestRegressor(
                                    n_estimators=1000,
                                    max_depth=7,
                                    max_features=0.2,
                                    max_leaf_nodes=100
)
mse = cross_val_score(clf, X_train, y_train,cv=5, scoring='neg_mean_squared_error').mean()
mse1 = math.sqrt(-1*mse)
print mse1
#
# #frame_sum.to_csv('train1.csv',index = None)


