#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T10%20-%202%20-%20Analisis%20de%20Componentes%20Principales%20SK%20Learn-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Análisis de Componentes Principales - SkLearn
# 

# In[6]:


get_ipython().system('pip install chart_studio')
import pandas as pd

import chart_studio.plotly as py
from plotly.graph_objs import * 
from chart_studio import tools as tls

from sklearn.preprocessing import StandardScaler

tls.set_credentials_file(username='JuanGabriel', api_key='6mEfSXf8XNyIzpxwb8z7')


# In[7]:


df = pd.read_csv("/content/python-ml-course/datasets/iris/iris.csv")


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

