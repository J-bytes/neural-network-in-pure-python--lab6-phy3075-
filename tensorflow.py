#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:36:54 2019

@author: jonathan
"""	
import pandas
import keras
from keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

nset=800
X=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12))[0:nset,:]                          # diverses instructions...
Y=np.loadtxt('trainingdata.txt',skiprows=1,usecols=(13))[0:nset]  
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(4, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	
	# Compile model
	model.compile(loss=losses.mean_squared_error, metrics=['accuracy'],optimizer='adam')
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=50, batch_size=2, verbose=2)
kfold = StratifiedKFold(n_splits=2, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))