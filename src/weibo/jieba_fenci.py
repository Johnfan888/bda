##!/usr/bin/env python
## coding=utf-8

#结巴分词
import jieba


filePath='/opt/lqu/packages/weibo/bowen_train.txt'
fileSegWordDonePath ='/opt/lqu/packages/weibo/bowen_train_fenci.txt'
# read the file by line
fileTrainRead = []
#fileTestRead = []
with open(filePath) as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(line)


# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print list[i],
# segment word with jieba
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    #分词
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i], cut_all=False)))])
    if i % 100 == 0 :
        print i


# to test the segment result
PrintListChinese(fileTrainSeg[10])

# save the result 分词结果
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n')