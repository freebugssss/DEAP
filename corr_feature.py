#输入一个m*n的矩阵，m个通道，n个采样点
#输出一个m*64的特征向量

import numpy as np
from sklearn import preprocessing
import scipy



def upper_right_triangle(mat):
    m,n=mat.shape
    res=[]
    if m!=n:
        print('m 不是方阵')
    for i in range(m):
        for j in range(i+1,m):
            res.append(mat[i,j])
    res=np.array(res)
    return res

def fft(time_data):
    return np.log10(np.absolute(np.fft.rfft(time_data,axis=1)[:,1:48]))

def freq_corr(fft_data):
    scaled=preprocessing.scale(fft_data,axis=0) #标准化
    corr_matrix=np.corrcoef(scaled) # 16*16
    eigenvalues=np.absolute(np.linalg.eig(corr_matrix)[0]) #1*16
    #eigenvalues.sorted()
    corr_coefficients=upper_right_triangle(corr_matrix)
    return np.concatenate((corr_coefficients,eigenvalues)) #1*136


def time_corr(data):
    resampled=scipy.signal.resample(data,400,axis=1) if data.shape[-1]>400 else data
    scaled=preprocessing.scale(resampled,axis=0)
    corr_matrix=np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    corr_coefficients = upper_right_triangle(corr_matrix)
    return np.concatenate((corr_coefficients, eigenvalues))

def transform(data):  #主函数
    fft_out=fft(data)
    freq_corr_out=freq_corr(fft_out)
    time_corr_out=time_corr(data)
    return np.concatenate((fft_out.ravel(),freq_corr_out,time_corr_out))

