#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('C:\\Users\\Anurag Bhoskar\\Desktop\\INR=X.csv')


# In[3]:


dataset.head()



# In[6]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[7]:


dataset.shape


# In[8]:


dataset.drop('Adj Close',axis = 1, inplace = True)


# In[9]:


dataset.head()


# In[10]:


dataset.isnull().sum()


# In[11]:


dataset.isna().any()


# In[12]:


dataset.info()


# In[13]:


dataset.describe()


# In[14]:


print(len(dataset))


# In[15]:


dataset['Open'].plot(figsize=(16,6))


# In[16]:


X  = dataset[['Open','High','Low','Volume']]
y = dataset['Close']


# In[17]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X ,y , random_state = 0)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()


# In[22]:


print(regressor.coef_)


# In[23]:


print(regressor.intercept_)


# In[24]:


predicted=regressor.predict(X_test)


# In[25]:


print(X_test)


# In[26]:


predicted.shape


# In[27]:


dframe=pd.DataFrame(y_test,predicted)


# In[28]:


dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})


# In[29]:


print(dfr)


# In[30]:


dfr.head(25)


# In[31]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[32]:


regressor.score(X_test,y_test)


# In[33]:


import math


# In[34]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted))


# In[35]:


print('Mean Squared  Error:',metrics.mean_squared_error(y_test,predicted))


# In[36]:


print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))


# In[37]:


graph=dfr.head(20)


# In[43]:


graph.plot(kind='bar')


# In[44]:


dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()


# In[46]:


dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))

