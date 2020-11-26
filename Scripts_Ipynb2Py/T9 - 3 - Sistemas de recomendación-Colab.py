#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T9%20-%203%20-%20Sistemas%20de%20recomendación-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Sistemas de recomendación
# ### Carga de datos de Movie Lens

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("/content/python-ml-course/datasets/ml-100k/u.data.csv", sep="\t", header=None)


# In[3]:


type(df)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns = ["UserID", "ItemID", "Rating", "TimeStamp"]


# In[7]:


df.head()


# ### Análisis exploratorio de los ítems

# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.hist(df.Rating)


# In[10]:


plt.hist(df.TimeStamp)


# In[11]:


df.groupby(["Rating"])["UserID"].count()


# In[12]:


plt.hist(df.groupby(["ItemID"])["ItemID"].count())


# ### Representación en forma matricial

# In[13]:


import numpy as np


# In[14]:


n_users = df.UserID.unique().shape[0]
n_users


# In[15]:


n_items = df.ItemID.unique().shape[0]
n_items


# In[16]:


ratings = np.zeros((n_users, n_items))


# In[17]:


for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]


# In[18]:


type(ratings)


# In[19]:


ratings.shape


# In[20]:


ratings


# In[21]:


sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
sparsity *= 100
print("Coeficiente de sparseidad: {:4.2f}%".format(sparsity))


# ### Crear conjuntos de entrenamiento y validación

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


ratings_train, ratings_test = train_test_split(ratings, test_size = 0.3, random_state=42)


# In[25]:


ratings_train.shape


# In[26]:


ratings_test.shape


# ## Filtro colaborativo basado en Usuarios
# * Matriz de similaridad entre los usuarios (distancia del coseno).
# * Predecir la valoración desconocida de un ítem *i* para un usuario activo *u* basandonos en la suma ponderada de todas las valoraciones del resto de usuarios para dicho ítem.
# * Recomendaremos los nuevos ítems a los usuarios según lo establecido en los pasos anteriores.

# In[27]:


import numpy as np
import sklearn


# In[28]:


sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_train)


# In[29]:


type(sim_matrix)


# In[30]:


sim_matrix.shape


# In[31]:


sim_matrix


# In[32]:


users_predictions = sim_matrix.dot(ratings_train) / np.array([np.abs(sim_matrix).sum(axis=1)]).T


# In[33]:


users_predictions


# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


def get_mse(preds, actuals):
    if preds.shape[0] != actuals.shape[0]:
        actuals = actuals.T
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)


# In[38]:


get_mse(users_predictions, ratings_train)


# In[39]:


sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_test)
users_predictions = sim_matrix.dot(ratings_test) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
get_mse(users_predictions, ratings_test)


# ## Filtro colaborativo basado en los KNN

# In[40]:


from sklearn.neighbors import NearestNeighbors


# In[41]:


k = 5


# In[42]:


neighbors = NearestNeighbors(k, 'cosine')


# In[43]:


neighbors.fit(ratings_train)


# In[44]:


top_k_distances, top_k_users = neighbors.kneighbors(ratings_train, return_distance=True)


# In[45]:


top_k_distances.shape


# In[46]:


top_k_distances[0]


# In[47]:


top_k_users.shape


# In[48]:


top_k_users[0]


# In[49]:


users_predicts_k = np.zeros(ratings_train.shape)
for i in range(ratings_train.shape[0]):# para cada usuario del conjunto de entrenamiento
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T


# In[50]:


users_predicts_k.shape


# In[51]:


users_predicts_k


# In[52]:


get_mse(users_predicts_k, ratings_train)


# In[54]:


users_predicts_k = np.zeros(ratings_test.shape)
for i in range(ratings_test.shape[0]):# para cada usuario del conjunto de test
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_test[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T
get_mse(users_predicts_k, ratings_test)


# In[55]:


ratings_test


# ## Filtro colaborativo basado en Items

# In[100]:


n_movies = ratings_train.shape[1]
n_movies


# In[101]:


neighbors = NearestNeighbors(n_movies, 'cosine')


# In[102]:


neighbors.fit(ratings_train.T)


# In[106]:


top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)


# In[104]:


top_k_distances.shape


# In[119]:


top_k_distances


# In[108]:


top_k_items.shape


# In[114]:


top_k_items


# In[109]:


item_preds = ratings_train.dot(top_k_distances) / np.array([np.abs(top_k_distances).sum(axis=1)])


# In[110]:


item_preds.shape


# In[111]:


item_preds


# In[112]:


get_mse(item_preds, ratings_train)


# In[113]:


get_mse(item_preds, ratings_test)


# ### Filtrado colaborativo basado en KNN

# In[120]:


k = 30
neighbors = NearestNeighbors(k, 'cosine')
neighbors.fit(ratings_train.T)
top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)


# In[121]:


top_k_distances.shape


# In[122]:


top_k_items[0]


# In[123]:


top_k_distances[0]


# In[139]:


preds = np.zeros(ratings_train.T.shape)
for i in range(ratings_train.T.shape[0]):
    if(i%50==0):
        print("iter "+str(i))
    den = 1
    if (np.abs(top_k_distances[i]).sum(axis=0)>0):
        den = np.abs(top_k_distances[i]).sum(axis=0)
    preds[i, :] = top_k_distances[i].dot(ratings_train.T[top_k_items][i])/np.array([den]).T


# In[158]:


get_mse(preds, ratings_train)


# In[159]:


get_mse(preds, ratings_test)


# In[152]:


preds.shape


# In[153]:


ratings_train.shape


# In[154]:


ratings_test.shape


# In[ ]:




