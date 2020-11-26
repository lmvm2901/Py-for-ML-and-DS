#!/usr/bin/env python
# coding: utf-8

# # Regresión lineal en Python
# ## El paquete scikit-learn para regresión lineal y la selección de rasgos

# In[1]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("../datasets/ads/Advertising.csv")


# In[3]:


feature_cols = ["TV", "Radio", "Newspaper"]


# In[4]:


X = data[feature_cols]
Y = data["Sales"]


# In[14]:


estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=2, step=1)
selector = selector.fit(X,Y)


# In[15]:


selector.support_


# In[16]:


selector.ranking_


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


X_pred = X[["TV", "Radio"]]


# In[19]:


lm = LinearRegression()
lm.fit(X_pred, Y)


# In[20]:


lm.intercept_


# In[21]:


lm.coef_


# In[22]:


lm.score(X_pred, Y)

