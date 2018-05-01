from __future__ import division
import os
import sys
import re
import scipy as sp
import numpy as np 
import math
import pandas
import matplotlib.pyplot as plt 
%matplotlib inline
from random import shuffle
from numpy import fft, cos, sin, pi, arange
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, freqz, kaiserord, firwin, hilbert, chirp
from scipy import arange , signal
from matplotlib import gridspec
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid,axes, show
# --- Optional Signal Processing library
import librosa
# --- Machine Learining
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD

#---------------- Load dataset (All) ---------------
train_path = "dataset_test/train/"
files = os.listdir(train_path)
j = 0
x_train = np.zeros((86,360000))
for wav in files:  
    signal, sr = librosa.load(train_path+wav)
    X = np.array(signal)
    Xpad = np.pad(X, (0, 360000-len(X)), mode='constant')
    x_train[j]=np.array(Xpad)
    j += 1

i=0
y_train = np.zeros((86,1))
for wav in files:    
    rt = int(wav[0:5])
    rt /=10000
    y_train[i] = rt
    i +=1
    


test_path = "dataset_test/test/"
files = os.listdir(test_path)

#Dataset 
j = 0
x_test = np.zeros((len(files),360000))
for wav in files:  
    signal, sr = librosa.load(test_path+wav)
    X = np.array(signal)
    Xpad = np.pad(X, (0, 360000-len(X)), mode='constant')
    x_test[j]=np.array(Xpad)
    j += 1
#Label    
i=0
y_test = np.zeros((len(files),1))
for wav in files:    
    rt = int(wav[0:5])
    rt /=10000
    y_test[i] = rt
    i += 1

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')    
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

#---------------- Normailized ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_norm= scaler.fit_transform(x_train)
y_train_norm = scaler.fit_transform(y_train)
x_test_norm= scaler.fit_transform(x_test)
y_test_norm = scaler.fit_transform(y_test)

#----------------- PCA ---------------
batch_size = 50
epochs = 1000
nPCA = 1000

pcaTrain = PCA(n_components=nPCA)
pcaTrain.fit(x_train_norm)
x_train_pca = pcaTrain.transform(x_train_norm)

pcaTest = PCA(n_components=nPCA)
pcaTest.fit(x_test_norm)
x_test_pca = pcaTest.transform(x_test_norm)

#-----------------Build Feed-forward NN ---------------
modelPCA = Sequential([
     Dense(32, input_shape=(nPCA,)),
     Activation('ReLu'),
     Dense(32, input_shape=(nPCA,)),
     Activation('ReLu'),
     Dense(32, input_shape=(nPCA,)),
     Activation('ReLu'),
     Dense(32, input_shape=(nPCA,)),
     Activation('relu'),
     Dense(16, input_shape=(nPCA,)),
     Activation('relu'),
     Dropout(0.5),
     Dense(1),
     Activation('linear'),
])
modelPCA.summary()
modelPCA.compile(optimizer='rmsprop',loss='mse',metrics=['mean_squared_error'])

#---------------- Training ---------------
modelPCA.fit(x_train_pca,y_train_norm,epochs=1000, batch_size=50,verbose=1)

#---------------- Save the model ---------------
model_json = modelPCA.to_json()
with open("model_nPCA900.json","w")as json_file:
     json_file.write(model_json)
modelPCA.save_weights("model_nPCA900.h5")
print("The new model is saved!")

#---------------- Evaluation ---------------
scores = modelPCA.evaluate(x_test_pca,y_test_norm)
yPredict = modelPCA(x_test_pca)

print("%s:%.3f%%"%(model.metrics_names[0],score[1]*100))
print("RMSE:%.3f",np.sqrt(metrics.mean_squared_error(y_test_norm,yPredict)))
