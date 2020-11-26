#!/usr/bin/env python
# coding: utf-8

# # Árboles de Regresión

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("../datasets/boston/Boston.csv")
data.head()


# In[3]:


data.shape


# In[4]:


colnames = data.columns.values.tolist()
predictors = colnames[:13]
target = colnames[13]
X = data[predictors]
Y = data[target]


# In[5]:


from sklearn.tree import DecisionTreeRegressor


# In[6]:


regtree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10, max_depth=5, random_state=0)


# In[7]:


regtree.fit(X,Y)


# In[8]:


preds = regtree.predict(data[predictors])


# In[9]:


data["preds"] = preds


# In[10]:


data[["preds", "medv"]]


# In[11]:


from sklearn.tree import export_graphviz
with open("resources/boston_rtree.dot", "w") as dotfile:
    export_graphviz(regtree, out_file=dotfile, feature_names=predictors)
    dotfile.close()
    
import os
from graphviz import Source
file = open("resources/boston_rtree.dot", "r")
text = file.read()
Source(text)


# In[ ]:


from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np


# In[ ]:


cv = KFold(n=X.shape[0], n_folds = 10, shuffle=True, random_state=1)
scores = cross_val_score(regtree, X, Y, scoring="mean_squared_error", cv = cv, n_jobs=1)
print(scores)
score = np.mean(scores)
print(score)


# In[ ]:


list(zip(predictors,regtree.feature_importances_))


# ## Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


forest = RandomForestRegressor(n_jobs=2, oob_score=True, n_estimators=10000)
forest.fit(X,Y)


# In[ ]:


data["rforest_pred"]= forest.oob_prediction_
data[["rforest_pred", "medv"]]


# In[ ]:


data["rforest_error2"] = (data["rforest_pred"]-data["medv"])**2
sum(data["rforest_error2"])/len(data)


# In[ ]:


forest.oob_score_

