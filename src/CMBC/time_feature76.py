# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:00:20 2018

@author: dell-1
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
'''
log = pd.read_csv('./data/sum_log.csv')
log = log.sort_values(['USRID','TIME'])

#统计每小时的点击数的各类值
log['TIME'] = pd.to_datetime(log['TIME'])
log['hour'] = log['TIME'].dt.hour
def timeToType(data):
    type_t = []
    for i in data:
        if 8<=i<12:
            type_t.append(1)
        elif 12<=i<17:
            type_t.append(2)
        elif 17<=i<22:
            type_t.append(3)
        else:
            type_t.append(4)
    return type_t
    
log['hour_type'] = timeToType(log['hour'])
del log['hour']
del log['OCC_TIM_SECONDS']
del log['dow']
del log['TCH_TYP']

log.to_csv('./data/time_click.csv',index=None)
'''

data = pd.read_csv('./data/time_click.csv',)
a = data.groupby(by=['USRID','DAY'])['hour_type'].value_counts().to_frame('click')
a = a.reset_index()
def date_range(step,end):
    if end-step < 0:
        print('date range not fit....')
    else:
        a = end-step
        b = end
        return a,b
'''
#=================================================
frame_final = a[['USRID','hour_type']]
for day in [1, 2, 3, 4, 7, 14]:
    begin,end = date_range(day,31)
    frame = a[(a['DAY'] >= begin) & (a['DAY']<= end)]
    frame1 = frame.groupby(['USRID','hour_type'], as_index=False)['click'].agg(
            {
                str(day) + 'day_type_mean': np.mean,
                str(day) + 'day_type_std': np.std,
                str(day) + 'day_type_min': np.min,
                str(day) + 'day_type_max': np.max,
             })
    frame_final = pd.merge(frame1,frame_final,on=['USRID','hour_type'],how='left')
frame_final.to_csv('./data/type_day1-14.csv',index=None)
#=============================================
'''
frame_final = a[['USRID']].drop_duplicates()
for day in [1, 2, 3, 4, 7, 14]:
    begin,end = date_range(day,31)
    frame = a[(a['DAY'] >= begin) & (a['DAY']<= end)]
    frame1 = frame.groupby(['USRID'], as_index=False)['click'].agg(
            {
                str(day) + 'day_mean': np.mean,
                str(day) + 'day_std': np.std,
                str(day) + 'day_min': np.min,
                str(day) + 'day_max': np.max,
             })
    frame_final = pd.merge(frame1,frame_final,on=['USRID'],how='left')

frame_final.to_csv('./data/day1-14.csv',index=None)
