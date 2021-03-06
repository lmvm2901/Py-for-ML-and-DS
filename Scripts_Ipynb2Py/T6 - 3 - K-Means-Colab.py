#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T6%20-%203%20-%20K-Means-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # El método de k-means

# In[1]:


import numpy as np


# In[2]:


data = np.random.random(90).reshape(30,3)
data


# In[3]:


c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))
clust_centers = np.vstack([data[c1], data[c2]])
clust_centers


# In[4]:


from scipy.cluster.vq import vq


# In[5]:


clusters = vq(data, clust_centers)
clusters


# In[6]:


labels = clusters[0]
labels


# In[7]:


get_ipython().system('pip install chart_studio')
import chart_studio.plotly as py
import plotly.graph_objects as go 
import plotly.graph_objects as go
import plotly.offline as ply


# In[ ]:


x = []
y = []
z = []
x2 = []
y2 = []
z2 = []

for i in range(0, len(labels)):
    if(labels[i] == 0):
        x.append(data[i,0])
        y.append(data[i,1])
        z.append(data[i,2])
        
    else:
        x2.append(data[i,0])
        y2.append(data[i,1])
        z2.append(data[i,2])

cluster1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.9
    ),
    name="Cluster 0"
)


cluster2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(127, 127, 127)',
        size=12,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    ),
    name="Cluster 1"
)
data2 = [cluster1, cluster2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=30
    )
)

fig = go.Figure(data=data2, layout=layout)
ply.plot(fig, filename='Clusters.html')


# In[8]:


from scipy.cluster.vq import kmeans


# In[9]:


kmeans(data, clust_centers)


# In[10]:


kmeans(data, 2)

