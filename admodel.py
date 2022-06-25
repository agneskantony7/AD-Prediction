#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas numpy')
import pandas as pd
import numpy as np


# In[2]:


dataframe = pd.read_csv('d:/Downloads/TADPOLE_D1_D2.csv')


# In[3]:


print(dataframe.head())


# In[4]:


print(dataframe.info())


# In[5]:


dataframe_subset = dataframe[['Ventricles', 'Hippocampus', 'WholeBrain','Fusiform', 'Entorhinal', 'MidTemp','RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting' ,'FDG','AGE','PTEDUCAT','APOE4','PTGENDER','MMSE']]
dataframe_subset.head()


# In[6]:


dataframe_subset=dataframe_subset.replace(np.nan,0)


# In[7]:


dataframe_subset= pd.get_dummies(dataframe_subset, columns = ['PTGENDER'])
print(dataframe_subset)


# In[8]:


import scipy.stats as stats
stats.zscore(dataframe_subset)


# In[9]:


dataframe_subset.info()


# In[10]:


x=dataframe_subset.drop(['MMSE'],axis=1).values
y=dataframe_subset['MMSE'].values


# In[11]:


print(x)


# In[12]:


print(y)


# In[13]:


from sklearn.svm import LinearSVC
#from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(x)
X_new.shape


# In[14]:


dataframe_subset.head()


# In[15]:


dataframe_subset.info()


# In[16]:


print(x)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[18]:


from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
x_train.shape, x_test.shape,y_test.shape


# In[19]:


sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_.fit(x_train, np.ravel(y_train,order='C'))
sel_.get_support()
x_train = pd.DataFrame(x_train)
print(x_train)


# In[20]:


selected_feat = x_train.columns[(sel_.get_support())]
print('total features: {}'.format((x_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
np.sum(sel_.estimator_.coef_ == 0)))


# In[21]:


from sklearn.linear_model import Lasso
m1=Lasso()
m1.fit(x_train,y_train)


# In[22]:


y_pred=m1.predict(x_test)
print(y_pred)


# In[23]:



m1.predict([[84599.0,5319.0,1129830.0,15506.0,1791.0,18422.0,22.0,1.0,4.0,100.0000,1.09079,81.3,18,1.0,0,1]])


# In[24]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[25]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')


# In[26]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[27]:


y_pred1=model.predict(x_test)
print(y_pred1)


# In[28]:


model.predict([[84599.0,5319.0,1129830.0,15506.0,1791.0,18422.0,22.0,1.0,4.0,100.0000,1.09079,81.3,18,1.0,0,1]])
model.predict([[84599.0,5319.0,1129830.0,15506.0,1791.0,18422.0,22.0,1.0,4.0,100.0000,1.09079,81.3,18,1.0,0,1]])


# In[29]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred1)


# In[30]:


# import the regressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0)

# fit the regressor with X and Y data
regressor.fit(x_train, y_train)


# In[31]:



y_pred2 = regressor.predict([[84599.0,5319.0,1129830.0,15506.0,1791.0,18422.0,22.0,1.0,4.0,100.0000,1.09079,81.3,18,1.0,0,1]])
y_pred3 = regressor.predict([[88580.0,5446.0,1100060.0,14400.0,2427.0,16972.0,19.0,2.0,6.0,100.0000,1.06360,81.3,18,1.0,0,1]])
y_pred4 = regressor.predict([[90099.0,5157.0,1095640.0,14617.0,1596.0,17330.0,31.0,2.0,7.0,100.0000,1.10384,81.3,18,1.0,0,1]])
print("Predicted price: % d\n"% y_pred2)
print("Predicted price: % d\n"% y_pred3)
print("Predicted price: % d\n"% y_pred4)


# In[35]:


# arange for creating a range of values
# from min value of X to max value of X
# with a difference of 0.01 between two
# consecutive values
X_grid = np.arange(min(x), max(x), 0.01)

# reshape for reshaping the data into
# a len(X_grid)*1 array, i.e. to make
# a column out of the X_grid values
X_grid = X_grid.reshape((len(X_grid), 1))

# scatter plot for original data
plt.scatter(x, y, color = 'red')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

# specify title
plt.title('Profit to Production Cost (Decision Tree Regression)')

# specify X axis label
plt.xlabel('Production Cost')

# specify Y axis label
plt.ylabel('Profit')

# show the plot
plt.show()


# In[ ]:


tree.plot_tree(regressor);


# In[33]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:



from sklearn import linear_model
MTL = linear_model.MultiTaskLasso(alpha = 0.5)
MTL.fit([[88580.0,5446.0,1100060.0,14400.0,2427.0,16972.0,19.0,2.0,6.0,100.0000,1.06360,81.3,18,1.0,0,1],[90099.0,5157.0,1095640.0,14617.0,1596.0,17330.0,31.0,2.0,7.0,100.0000,1.10384,81.3,18,1.0,0,1]])
print("Prediction result: \n", MTL.predict([[84599.0,5319.0,1129830.0,15506.0,1791.0,18422.0,22.0,1.0,4.0,100.0000,1.09079,81.3,18,1.0,0,1]]), "\n")
print("Coefficients: \n", MTL.coef_, "\n")

# print the intercepts
print("Intercepts: \n", MTL.intercept_, "\n")

# print the number of iterations performed
print("Number of Iterations: ", MTL.n_iter_, "\n")

