import pickle
import keras
f=open('s01.dat','rb')
a=pickle.load(f,encoding='latin1')
f.close()

print(a)

"第二个版本"