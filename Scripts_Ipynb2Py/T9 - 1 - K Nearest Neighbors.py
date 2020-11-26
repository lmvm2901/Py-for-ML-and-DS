#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors

# In[1]:


import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


# In[76]:


df = pd.read_csv("../datasets/cancer/breast-cancer-wisconsin.data.txt", header=None)

df.head()


# In[77]:


df.describe()


# In[78]:


df.columns = ["name", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "class"]


# In[79]:


df.head()


# In[80]:


df = df.drop(["name"],1)


# In[81]:


df.replace("?", -99999, inplace=True)


# In[82]:


Y = df["class"]


# In[83]:


X = df[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]]


# In[84]:


X.head()


# In[85]:


Y.head()


# ## Clasificador de los K vecinos 

# In[86]:


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)


# In[87]:


clf = neighbors.KNeighborsClassifier()


# In[88]:


clf.fit(X_train, Y_train)


# In[89]:


accuracy = clf.score(X_test, Y_test)
accuracy


# # Clasificación sin limpieza

# In[75]:


df = pd.read_csv("../datasets/cancer/breast-cancer-wisconsin.data.txt", header=None)
df.replace("?", -99999, inplace=True)
df.columns = ["name", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "class"]

Y = df["class"]
X = df[["name", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
accuracy


# # Clasificar nuevos datos

# In[90]:


sample_measure = np.array([4,2,1,1,1,2,3,2,1])


# In[92]:


sample_measure = sample_measure.reshape(1,-1)


# In[93]:


predict = clf.predict(sample_measure)


# In[94]:


predict


# In[113]:


sample_measure2 = np.array([[4,2,1,1,1,2,3,2,1], [2,2,4,4,2,2,6,2,4]]).reshape(2,-1)


# In[114]:


predict = clf.predict(sample_measure2)


# In[115]:


predict


# In[ ]:




