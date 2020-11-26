#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T6%20-%201%20-%20Distancias-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Distancias

# In[1]:


from scipy.spatial import distance_matrix
import pandas as pd


# In[2]:


data = pd.read_csv("/content/python-ml-course/datasets/movies/movies.csv", sep=";")
data


# In[3]:


movies = data.columns.values.tolist()[1:]
movies


# In[4]:


dd1 = distance_matrix(data[movies], data[movies], p=1)
dd2 = distance_matrix(data[movies], data[movies], p=2)
dd10 = distance_matrix(data[movies], data[movies], p=10)


# In[5]:


def dm_to_df(dd, col_name):
    import pandas as pd
    return pd.DataFrame(dd, index=col_name, columns=col_name)


# In[6]:


dm_to_df(dd1, data["user_id"])


# In[7]:


dm_to_df(dd2, data["user_id"])


# In[8]:


dm_to_df(dd10, data["user_id"])


# In[9]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[10]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs = data["star_wars"], ys = data["lord_of_the_rings"], zs=data["harry_potter"])


# # Enlaces

# In[11]:


df = dm_to_df(dd1, data["user_id"])
df


# In[12]:


Z=[]


# In[13]:


df[11]=df[1]+df[10]
df.loc[11]=df.loc[1]+df.loc[10]
Z.append([1,10,0.7,2])#id1, id2, d, n_elementos_en_cluster -> 11.
df


# In[14]:


for i in df.columns.values.tolist():
    df.loc[11][i] = min(df.loc[1][i], df.loc[10][i])
    df.loc[i][11] = min(df.loc[i][1], df.loc[i][10])
df


# In[15]:


df = df.drop([1,10])
df = df.drop([1,10], axis=1)
df


# In[16]:


x = 2
y = 7

n = 12

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],2])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis=1)
df


# In[17]:


x = 5
y = 8

n = 13

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],2])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis=1)
df


# In[18]:


x = 11
y = 13

n = 14

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],2])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis=1)
df


# In[19]:


x = 9
y = 12
z = 14

n = 15

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],3])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][z])

df = df.drop([x,y,z])
df = df.drop([x,y,z], axis=1)
df


# In[20]:


x = 4
y = 6
z = 15

n = 16

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],3])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][z])

df = df.drop([x,y,z])
df = df.drop([x,y,z], axis=1)
df


# In[21]:


x = 3
y = 16

n = 17

df[n]=df[x]+df[y]
df.loc[n]=df.loc[x]+df.loc[y]
Z.append([x,y,df.loc[x][y],2])#id1, id2, d, n_elementos_en_cluster -> 11.

for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])

df = df.drop([x,y])
df = df.drop([x,y], axis=1)
df


# In[22]:


Z


# # Clustering jerárquico

# In[23]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# In[24]:


movies


# In[25]:


data[movies]


# In[26]:


Z = linkage(data[movies], "ward")
Z
plt.figure(figsize=(25,10))
plt.title("Dendrograma jerárquico para el Clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=10.0)
plt.show()


# In[27]:


Z = linkage(data[movies], "average")
Z
plt.figure(figsize=(25,10))
plt.title("Dendrograma jerárquico para el Clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=10.0)
plt.show()


# In[28]:


data[movies]


# In[29]:


Z = linkage(data[movies], "complete")
Z
plt.figure(figsize=(25,10))
plt.title("Dendrograma jerárquico para el Clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=10.0)
plt.show()


# In[30]:


Z = linkage(data[movies], method="single", metric="cosine")
Z
plt.figure(figsize=(25,10))
plt.title("Dendrograma jerárquico para el Clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=10.0)
plt.show()


# The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
