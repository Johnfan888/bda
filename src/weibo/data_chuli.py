##!/usr/bin/env python
## coding=utf-8
import csv
from datetime import datetime
import collections
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# 处理原始数据
#1. 将时间转换为周几，和 几点
#2. 将userID 单独提取，做embedding
#3. 将博文内容提取，提取关键词并embedding
outfile = open(r"E:\weibo\user_final.txt", 'w')
try:
    file1 = open("E:\weibo\weibo_train_data.txt", 'r')
    file2 = open("E:\weibo\weibo_predict_data.txt", 'r')
    for line in file1:   # train文件
        data = line.split("\t")
        outfile.write(data[0])
        # t = datetime.strptime(data[2], "%Y-%m-%d %H:%M:%S")
        # outfile.write(data[0]+"\t"+data[1]+"\t")
        # outfile.write(str(t.weekday()))
        # outfile.write("\t")
        # outfile.write(str(t.hour))
        # outfile.write("\t"+data[3]+"\t"+data[4]+"\t"+data[5])
        outfile.write("\n")
    for line in file2:  # predict文件
        data = line.split("\t")
        outfile.write(data[0])
        # t = datetime.strptime(data[2], "%Y-%m-%d %H:%M:%S")   # 时间处理
        # outfile.write(data[0] + "\t" + data[1] + "\t")
        # outfile.write(str(t.weekday()))
        # outfile.write("\t")
        # outfile.write(str(t.hour))
        # outfile.write("\t" + data[3] + "\t" + data[4] + "\t" + data[5])
        outfile.write("\n")



        # data_dict = {data[2]:(data[0],data[1],data[3]+data[4]+data[5],data[6])}
        # outfile.write(data[3])
        # outfile.write(",")
        # str1 = "".join(data[3])
        #
        # outfile.write(len(str))
        # outfile.write("\n")
except IOError as err:
    print('File error: ' + str(err))
finally:
    if 'file' in locals():
        file.close()
outfile.close()
#
# # 查找predication中的userID是否存在与train文件的userID中
# count_shuyu = []
# count_bushuyu = []
# for str in list2:
#     if str in list1:
#         count_shuyu.append("True")
#     else:
#         count_bushuyu.append("False")
#
# print(len(count_shuyu))
# print(len(count_bushuyu))

# for i, val in enumerate(list_a):
#     txtName ="E:\WeiboData\wenzhang\\"+str(i)+".txt"
#     outfile = open(txtName, 'w')
#     outfile.write(list_a[i])
# length = []
# for str1 in list_a:
#     length.append(len(str1))
#
# print max(length)

