##!/usr/bin/env python
## coding=utf-8
#1. 获取关键词
#2. 如果关键字小于5个，用0补上
import jieba
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

in_file = "/opt/lqu/packages/weibo/bowen_test.txt"
out_file = open("/opt/lqu/packages/weibo/bowen_final.txt","a")
# 精确模式 
with open(in_file,"r") as fr:
    for line in fr:
        # 获取关键词
        tags1 = jieba.analyse.textrank(line,topK=5)              # return list
        tags2 = jieba.analyse.extract_tags(line,topK=5)    # tf-idf :  return list
        # tags = tags1+tags2
        # tags = list(set(tags))

        if tags1:
            for str1 in tags1:
                out_file.write(str1 + " ")
            if len(tags1) != 5:
               for i in range(5-len(tags1)):
                   out_file.write("0"+" ")
            out_file.write("\n") 
        else:
            for str2 in tags2:
                out_file.write(str2 + " ")
            if len(tags2) != 5:
                for i in range(5-len(tags2)):
                    out_file.write("0" + " ")
            out_file.write("\n")


        # out_file.write("---------------")
        # for str1 in tags1:
        #     out_file.write(str1 + " ")
        #     out_file.write(",")
        # for str1 in tags2:
        #     out_file.write(str1 + " ")
        # out_file.write("\n")


out_file.close()


