# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:23:39 2018

@author: dell-1
"""
import pandas as pd
import numpy as np
import datetime

def get_sum_log():
    #合并train_log和test_log表
    train_log = pd.read_csv('./data/train_log.csv',sep='\t')
    test_log = pd.read_csv('./data/test_log.csv',sep='\t')
    log = pd.concat([train_log,test_log],copy=False)

    #将OCC_TIME转为时间类型，并按照（USERID，TIME）的从小到大排序
    log['TIME'] = pd.to_datetime(log['OCC_TIM'])
    log['DAY'] = log['TIME'].dt.day     #3月几号
    log['dow'] = log['TIME'].dt.dayofweek   #周几
    log = log.sort_values(['USRID','TIME'])
    #将EVT_LBL字段分开
    log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x:x.split('-')[0])
    log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])
    log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x:x.split('-')[2])
    del log['EVT_LBL']
    del log['TIME']
    return log

def get_EVT(log):
    def get_top(str1):
        #对EVT统计，选出top n个作为属性
        #EVT_LBL_1选取所有出现的作为属性，EVT_LBL_2和3选取前10个作为属性
        first_EVT = log[str1].value_counts().reset_index()
        first_EVT.columns = [str1,'count']
        if(str1 == 'EVT_LBL_1'):
            first_EVT = first_EVT
        else:
            first_EVT = first_EVT.head(10)
        #列名
        cols = [str1+'_%d' % column for column in range(1,len(first_EVT)+1)]
        #统计每个用户每个类别的点击次数,转换为Dataframe
        a = log.groupby(by=['USRID'])[str1].value_counts().to_frame('EVT')
        EVT1 = pd.Series(a.values.reshape(len(a)),index=a.index) 
        EVT1 = EVT1.unstack()
        #选取top列
        need_col = [column for column in  first_EVT[str1] ]
        EVT1 = EVT1[need_col]
        #更改列名
        d = dict(zip(first_EVT[str1],cols))
        #    print(d)
        EVT1.rename(columns=d, inplace = True)   
        return EVT1
    EVT1_top  = get_top('EVT_LBL_1')
    EVT2_top  = get_top('EVT_LBL_2')
    EVT3_top  = get_top('EVT_LBL_3')
    EVT = pd.merge(EVT1_top,EVT2_top,left_index=True,right_index=True)
    EVT = pd.merge(EVT,EVT3_top,left_index=True,right_index=True)
    EVT = EVT.reset_index()
    return EVT

def get_click(log):
    #统计用户在一个月之内点击多少次，并计算周一至周日各点击多少次
    user = log['USRID'].value_counts().reset_index()
    user.columns = ['USRID','click_num']
    #该用户day_of_week点击几次
    a = log.groupby(by=['USRID'])['dow'].value_counts().to_frame('day_click')
    user_day_click = pd.Series(a.values.reshape(len(a)),index=a.index) 
    user_day_click = user_day_click.unstack()
    user_day_click.columns = ['day7_click','day1_click','day2_click','day3_click','day4_click',
                          'day5_click','day6_click',]

    user_day_click['USRID'] = user_day_click.index
    user_day_click = user_day_click.reset_index(drop=True)

    user = pd.merge(user,user_day_click,on = ['USRID'])
    return user
    
def get_click_interval(log):   
#计算每个用户的相邻两次之间的间隔，并求这些间隔的最大，最小，均值，中值
#如果该用户只有一条点击记录，赋值-1   
    def sec_diff(a,b):
        if (a is np.nan) | (b is np.nan):
            return -1
        return (datetime.datetime.strptime(str(b), "%Y-%m-%d %H:%M:%S")-datetime.datetime.strptime(str(a), "%Y-%m-%d %H:%M:%S")).seconds
 
    
    usr_click_diff = []
    for name, group in log.groupby('USRID'):
        tmp = group['OCC_TIM'].values
        diff_usr = []
        if len(tmp) == 1:
            usr_click_diff.append([name,-1,-1,-1,-1])
        else:
            for i in range(0,len(tmp)-1):
                diff_usr.append(sec_diff(tmp[i+1],tmp[i]))
            max_diff=np.max(diff_usr)
            min_diff=np.min(diff_usr)
            avg_diff=np.mean(diff_usr)
            mid_diff=np.median(diff_usr)
            usr_click_diff.append([name,max_diff,min_diff,avg_diff,mid_diff])
    usr_click_interval = pd.DataFrame(usr_click_diff,
                                      columns = ['USRID','max_diff','min_diff','avg_diff','mid_diff'])
    return usr_click_interval
    
    
def get_day_click(log):
    #用户玩app的次数最多的那一天以及点击次数
    #用户玩app的次数最少的那一天以及点击次数
    a = log.groupby(by=['USRID'])['DAY'].value_counts().to_frame('most_click')
    a = a.reset_index()
    most_Day_Click = a.sort_values('most_click', ascending=False).groupby('USRID', as_index=False).first()
    most_Day_Click.columns = ['USRID','most_click_day','most_click']

    min_Day_Click = a.sort_values('most_click', ascending=True).groupby('USRID', as_index=False).first()
    min_Day_Click.columns = ['USRID','mininum_click_day','mininum_click']
    return most_Day_Click,min_Day_Click
    
    
#=================================================================  
log = get_sum_log()
usr_top_click = get_EVT(log)
usr_week_click = get_click(log)
usr_click_interval = get_click_interval(log)
most_Day_Click,min_Day_Click = get_day_click(log)

data = pd.merge(usr_top_click,usr_week_click, on = ['USRID'],how = 'left')
data = pd.merge(data,usr_click_interval, on = ['USRID'],how = 'left')
data = pd.merge(data,most_Day_Click, on = ['USRID'],how = 'left')
data = pd.merge(data,min_Day_Click, on = ['USRID'],how = 'left')
data =data.fillna(0)
