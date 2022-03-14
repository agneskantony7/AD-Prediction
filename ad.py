import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import Quandl
import math

dataframe = pd.read_csv('d:/Downloads/TADPOLE_D1_D2.csv')
print(dataframe.shape)
#null values
print(dataframe.isna().sum())
#drop the column with all missing values
dataframe = dataframe.dropna(axis=1)
print(dataframe.shape)
#print(dataframe['PTETHCAT'].value_counts())
print(dataframe.dtypes)
print(dataframe.head())
from sklearn import preprocessing
import numpy as np

a = np.random.random((1, 1000))
#a = a*20
#print("Data = ", a)

# normalize the data attributes
normalized = preprocessing.normalize(a)
print("Normalized Data = ", normalized)
#tadpole = pd.read_csv("d:/Downloads/TADPOLE_D1_D2.csv")
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(dataframe,train_size=0.8)

from sklearn.linear_model import LinearRegression
X = train_data[['SITE']]
y=train_data.RID
#build the linear regression object
lm=LinearRegression()
# Train the model using the training sets
lm.fit(X,y)
#print the y intercept
print(lm.intercept_)
#print the coefficents
print(lm.coef_)
print(dataframe.describe())
forcast_col = 'MMSE_bl'
dataframe['label']=dataframe[forcast_col]
print(dataframe.head())
print(dataframe.nunique())
#print(dataframe['MMSE_bl'])
