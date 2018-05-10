import word2vec
# 1. 将转换好的词向量文件，和博文关键词做对比，生成有序的词向量
# （一行）5个主题词--》50X5 = 250维

infile = '/root/word2vec/data/bowen_1.bin'
infile_2 = '/root/word2vec/data/bowen_final.txt'
outfile =  open('/root/word2vec/data/bowen_vectors.txt','w')
def transfer(infile,infile_2,outfile):
    d = {}
    with open(infile,"r") as fr:
        for line in fr:
            data = line.strip().split(" ")
            key = data[0]
            value = data[1:]
            d[key] = value

    with open(infile_2,"r") as fr1:
        for line in fr1:
            data1 = line.strip().split(" ")
            for i in data1:
                if d.has_key(i):
                    outfile.write(str(d[i]))
            outfile.write("\n")
    outfile.close()

if __name__ == "__main__":
    transfer()


