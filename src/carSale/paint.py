# -- coding:utf-8 --
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    return frame_sum
frame_sum = get_train()
frame_sum = frame_sum.reset_index()
x1 = frame_sum.columns.tolist()
x1 = x1[1:]


dbpath ='/root/word2vec/carSale/data/pic/%s.jpg'
for i in range(0,len(frame_sum)):
    y1 = frame_sum.iloc[i, 1:].as_matrix()
    a = []
    b = []
    for j in range(0, len(x1)):
        if y1[j] > 0:
            a.append(str(x1[j]))
            b.append(y1[j])
    plt.plot(a, b, 'o-',label=frame_sum.iloc[i,0])
    str1 = 'car'+str(frame_sum.iloc[i,0])
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    # # plt.show()
    plt.savefig(dbpath % str1)
    plt.close()





