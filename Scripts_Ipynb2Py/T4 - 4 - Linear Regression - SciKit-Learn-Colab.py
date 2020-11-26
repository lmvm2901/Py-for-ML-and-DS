#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T4%20-%204%20-%20Linear%20Regression%20-%20SciKit-Learn-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Regresión lineal en Python
# ## El paquete scikit-learn para regresión lineal y la selección de rasgos

# In[1]:


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("/content/python-ml-course/datasets/ads/Advertising.csv")


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

