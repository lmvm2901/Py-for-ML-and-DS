#!/usr/bin/env python
# coding: utf-8

# # Análisis de Componentes Principales - SkLearn
# 

# In[6]:


import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import * 
import plotly.tools as tls

from sklearn.preprocessing import StandardScaler

tls.set_credentials_file(username='JuanGabriel', api_key='6mEfSXf8XNyIzpxwb8z7')


# In[7]:


df = pd.read_csv("../datasets/iris/iris.csv")


# In[8]:


X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
X_std = StandardScaler().fit_transform(X)


# In[9]:


from sklearn.decomposition import PCA as sk_pca


# In[11]:


acp = sk_pca(n_components=2)
Y = acp.fit_transform(X_std)


# In[12]:


Y


# In[15]:


results = []

for name in ('setosa', 'versicolor', 'virginica'):
    result = go.Scatter (x = Y[y==name,0], y = Y[y==name,1],
    mode = "markers", name = name, marker = {"size":8, "line": {"color": "rgba(225,225,225,0.2)","width": 0.5}}, opacity= 0.75)
    results.append(result)

layout = go.Layout(xaxis = {"title":'CP1', "showline" :False, "zerolinecolor" : "gray"}, yaxis = {"title" :'CP2', "showline" :False, "zerolinecolor" : "gray"})

fig = go.Figure(data=results, layout=layout)
py.iplot(fig)
#fig.show()


# In[ ]:




