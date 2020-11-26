#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T6%20-%204%20-%20Clustering%20completo-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Clustering con Python

# ### Importar el dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("/content/python-ml-course/datasets/wine/winequality-red.csv", sep = ";")
df.head()


# In[3]:


df.shape


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.hist(df["quality"])


# In[6]:


df.groupby("quality").mean()


# ### Normalización de los datos

# In[7]:


df_norm = (df-df.min())/(df.max()-df.min())
df_norm.head()


# ## Clustering jerárquico con scikit-learn

# In[8]:


from sklearn.cluster import AgglomerativeClustering


# In[9]:


clus= AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df_norm)


# In[10]:


md_h = pd.Series(clus.labels_)


# In[11]:


plt.hist(md_h)
plt.title("Histograma de los clusters")
plt.xlabel("Cluster")
plt.ylabel("Número de vinos del cluster")


# In[12]:


clus.children_


# In[13]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[14]:


Z = linkage(df_norm, "ward")


# In[15]:


plt.figure(figsize=(25,10))
plt.title("Dendrograma de los vinos")
plt.xlabel("ID del vino")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=4.)
plt.show()


# ## K-means

# In[16]:


from sklearn.cluster import KMeans
from sklearn import datasets


# In[17]:


model = KMeans(n_clusters=6)
model.fit(df_norm)


# In[18]:


model.labels_


# In[19]:


md_k = pd.Series(model.labels_)


# In[20]:


df_norm["clust_h"] = md_h
df_norm["clust_k"] = md_k


# In[21]:


df_norm.head()


# In[22]:


plt.hist(md_k)


# In[23]:


model.cluster_centers_


# In[24]:


model.inertia_


# ## Interpretación final

# In[25]:


df_norm.groupby("clust_k").mean()

