# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:47:50 2018

@author: dell-1
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import time

def base_model(data):
    train = data[data['FLAG']!=-1]
    test = data[data['FLAG']==-1]

    train_userid = train.pop('USRID')
    y = train.pop('FLAG')
    col = train.columns
    X = train[col].values

    test_userid = test.pop('USRID')
    test_y = test.pop('FLAG')
    test = test[col].values

    N = 5
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=4396)

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    xx_cv = []
    xx_pre = []

    for train_in,test_in in skf.split(X,y):
        X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': {'auc'},
            'num_leaves': 32,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
            }
            
        print('Start training...')
            # train
        gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

        print('Start predicting...')
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        xx_cv.append(roc_auc_score(y_test,y_pred))
        xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

    s = 0
    for i in xx_pre:
        s = s + i

    s = s /N

    res = pd.DataFrame()
    res['USRID'] = list(test_userid.values)
    res['RST'] = list(s)
    
    a = pd.DataFrame({
        'column': col,
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance')

    print(a)
    print('xx_cv',np.mean(xx_cv))
    
    
    return res,xx_cv,a
    
def submit(res,xx_cv):
    time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    res.to_csv('./submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')
    print('finished!!!')
