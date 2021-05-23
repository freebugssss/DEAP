# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

df=pd.read_csv('./dataset/dataset.csv',header=0, index_col=0)
dataset=df.values
X=dataset[:,:-1]
Y=dataset[:,-1]
Y=keras.utils.to_categorical(Y,4)

X, test_X, Y, test_y = train_test_split(X, Y, test_size=0.2, stratify=Y)

input_shape_X=X.shape[-1]
input_shape_Y=Y.shape[-1]

print('X:',X.shape,"Y:",Y.shape)
# create model
model = Sequential()
model.add(Dense(10240, input_shape=(input_shape_X,),activation='relu',kernel_initializer="normal"))

model.add(Dense(5120, kernel_initializer='normal', activation='relu'))
model.add(Dense(input_shape_Y, kernel_initializer='normal', activation='sigmoid'))
# Compile model
#adam=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history=model.fit(X, Y, epochs=150, batch_size=30, verbose=2,shuffle=True,validation_data=(test_X, test_y))
# evaluate the model
#scores = model.evaluate(X_test, Y_test)
#print("AAA%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
