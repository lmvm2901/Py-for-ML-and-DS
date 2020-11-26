#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T7%20-%202%20-%20Trees%20-%20Árboles%20de%20Regresión-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Clonamos el repositorio para obtener los dataSet

# In[ ]:


get_ipython().system('git clone https://github.com/joanby/python-ml-course.git')


# # Damos acceso a nuestro Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
# Test it
get_ipython().system("ls '/content/drive/My Drive' ")


# In[ ]:


from google.colab import files # Para manejar los archivos y, por ejemplo, exportar a su navegador
import glob # Para manejar los archivos y, por ejemplo, exportar a su navegador
from google.colab import drive # Montar tu Google drive


# # Árboles de Regresión

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("/content/python-ml-course/datasets/boston/Boston.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


colnames = data.columns.values.tolist()
predictors = colnames[:13]
target = colnames[13]
X = data[predictors]
Y = data[target]


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


regtree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10, max_depth=5, random_state=0)


# In[ ]:


regtree.fit(X,Y)


# In[ ]:


preds = regtree.predict(data[predictors])


# In[ ]:


data["preds"] = preds


# In[ ]:


data[["preds", "medv"]]


# In[ ]:


from sklearn.tree import export_graphviz
with open("/content/python-ml-course/notebooks/resources/boston_rtree.dot", "w") as dotfile:
    export_graphviz(regtree, out_file=dotfile, feature_names=predictors)
    dotfile.close()
    
import os
from graphviz import Source
file = open("/content/python-ml-course/notebooks/resources/boston_rtree.dot", "r")
text = file.read()
Source(text)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np


# In[ ]:


cv = KFold(n_splits = 10, shuffle=True, random_state=1)
cv.get_n_splits(X)
scores = cross_val_score(regtree, X, Y, scoring=make_scorer(mean_squared_error), cv = cv, n_jobs=1)
print(scores)
score = np.mean(scores)
print(score)


# In[ ]:


list(zip(predictors,regtree.feature_importances_))


# ## Random Forests

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


forest = RandomForestRegressor(n_jobs=-1, oob_score=True, n_estimators=10000)
forest.fit(X,Y)


# In[ ]:


data["rforest_pred"]= forest.oob_prediction_
data[["rforest_pred", "medv"]]


# In[ ]:


data["rforest_error2"] = (data["rforest_pred"]-data["medv"])**2
sum(data["rforest_error2"])/len(data)


# In[ ]:


forest.oob_score_

