##!/usr/bin/env python
## coding=utf-8
import numpy
import word2vec
#将词语转换为词向量
word2vec.word2vec('/root/word2vec/b1.txt',
                  '/root/word2vec/b1.bin',
                  size=50, verbose=True,min_count=1,binary=0)
model = word2vec.load('/root/word2vec/b1.bin')

# print model.vectors.shape
# print type(model.vectors)
# outfile = open('/root/word2vec/bowen_test_vector.txt','w')
numpy.savetxt('/root/word2vec/b1-1.txt',model.vectors)

#
# for index in indexes[0]:
#     print (model.vocab[index])




# for i in range(5):
#     print model.vocab[i]
# print model.vectors
# outfile = open('/root/word2vec/bowen_test_vector.txt','w')
# outfile.write(model.vectors)

#
# print model[u'扣响'][:]
# # print model['dog'][:]
# indexes, metrics = model.cosine('')

# index = -1
# print model.vocab[index]

# print model.generate_response(indexes, metrics)
# print  model.generate_response(indexes, metrics).tolist()

# print model["anarchism"].shape