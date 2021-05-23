import pickle
import keras
import pandas as pd
import corr_feature
import numpy as np
from sklearn.utils import shuffle
import os

slice_interval = 2  # chosen 2 second slice(2*128)
begin_time = 10  # 选择脑电开始截取的时间点，单位：秒，0~62
frequency = 128  # 采样频率

file_names=os.listdir('./data_preprocessed_python')
df_all = pd.DataFrame(columns=list(range(2561)))


def judge_label(valence_mean, arouse_mean, valence, arouse): #判断样本的种类
    if valence >= valence_mean and arouse >= arouse_mean:
        return 0
    elif valence >= valence_mean and arouse < arouse_mean:
        return 1
    elif valence < valence_mean and arouse >= arouse_mean:
        return 2
    else:
        return 3

for i in file_names:  #遍历所有的数据集
    f=open('./data_preprocessed_python/'+i,'rb')
    a=pickle.load(f,encoding='latin1')
    f.close()
    print('Successful loaded '+i + '. Processing...')
    data=a['data']  #40*40*8064
    label=a['labels']  #40*4


    p_data=[]
    p_label=[]
    valence_mean=np.mean(label[:,0])
    arouse_mean=np.mean(label[:,1])

    for trials in range(len(data)):
        for slice in range(begin_time,62):
            time_slice=data[trials,:32,slice*frequency:(slice+slice_interval)*128]
            feature=corr_feature.transform(time_slice)
            p_data.append(feature)
            p_label.append(judge_label(valence_mean,arouse_mean,label[trials,0],label[trials,1]))

    p_data=np.array(p_data)
    p_label=np.array(p_label)
    df=np.c_[p_data,p_label] #拼接
    df=pd.DataFrame(df)
    df_all=pd.merge(df_all,df,how='outer')
    print(i+' is processed')


df_all=shuffle(df_all)
df_all.to_csv('./dataset/dataset.csv')









