''' @ Author Vishnu Varthan, Oct 6,2018'''
#Required Dependencies
import pandas as pd
import math
import quandl
import numpy as np


''' Data Preprocessing '''
df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close','Adj. Volume']]


df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-999,inplace=True)


forecast_out = int(math.ceil(0.01 * len(df)))
label = df[forecast_col].shift(-forecast_out)


''' In this model we are predicting the percentage increase based on the Features'''
df.dropna(inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score
x=np.array(df.drop([label],1,inplace=True))
y=np.array(df[label])
''' Cleaning data removing headers and the other stuffs and using it suitable for computation
 '''
x=preprocessing.scale(x)
x=x[:-forecast_out]
x_train,y_train,x_test,y_test=train_test_split(x,y)
clf=LinearRegression()
clf.fit(x_train,y_train)
confidence=clf.score(x_train,y_train)
print("The accuracy of model is {}".format(confidence))