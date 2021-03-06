# -*- coding: utf-8 -*-
"""Keras_edx


#Test
import keras

''' Author @Vishnu Varthan
Regression model in keras'''
import pandas as pd
import numpy as np
data=pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
data.describe()

#Checking if some vals are missing and summing the missing places in each cols


data.isnull().sum()



#Seperating the column names seperately for indexing
data_cols=data.columns


#Seperating Datas
predictor=data[data_cols[data_cols!='Strength']]
#print(predictor)
target=data['Strength']
#Testing by printing
predictor.head(5)
#Check ok!!!

#We are normalizing the data set for easy computation
predictor_simple=(predictor-predictor.mean())/predictor.std()

#Defines the input shape in this case 8 so 8 input perceptrons will be there ie no of cols
cols=predictor_simple.shape[1]
print(cols)
'''Now the actual stuff'''
from keras.models import Sequential
from keras.layers import Dense


'''Regression model from scratch'''
  
  #Construct the structure of Neural Networks
  
  
model=Sequential()

#Adding first layer
model.add(Dense(100,activation='relu',input_shape=(cols,)))

#Adding Second layer
model.add(Dense(50,activation='relu'))
model.add(Dense(1))
  #Intitialize loss function,optimizer etc...
model.compile(optimizer='adam',loss='mean_squared_error')
'''Ready'''

model.fit(predictor_simple,target,epochs=100,verbose=2,validation_split=0.5)



