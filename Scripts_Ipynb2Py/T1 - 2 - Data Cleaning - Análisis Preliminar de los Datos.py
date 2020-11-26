#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T1%20-%202%20-%20Data%20Cleaning%20-%20An%C3%A1lisis%20Preliminar%20de%20los%20Datos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# # Resumen de los datos: dimensiones y estructuras

# In[ ]:


import pandas as pd
import os


# In[ ]:


mainpath = "/content/drive/My Drive/Curso Machine Learning con Python/datasets"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)

urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"


# In[ ]:


data = pd.read_csv(urldata)


# In[5]:


data.head(10)


# In[6]:


data.tail(8)


# In[7]:


data.shape


# In[8]:


data.columns.values


# Vamos a hacer un resumen de los estadísticos básicos de las variables numéricas.

# In[9]:


data.describe()


# In[10]:


data.dtypes


# # Missing values

# In[11]:


pd.isnull(data["body"])


# In[12]:


pd.notnull(data["body"])


# In[13]:


pd.isnull(data["body"]).values.ravel().sum()


# In[14]:


pd.notnull(data["body"]).values.ravel().sum()


# Los valores que faltan en un data set pueden venir por dos razones:
# * Extracción de los datos
# * Recolección de los datos

# #### Borrado de valores que faltan

# In[15]:


data.dropna(axis=0, how="all")


# In[ ]:


data2 = data


# In[17]:


data2.dropna(axis=0, how="any")


# #### Cómputo de los valores fantantes

# In[ ]:


data3 = data


# In[19]:


data3.fillna(0)


# In[ ]:


data4 = data


# In[ ]:


data4= data4.fillna("Desconocido")


# In[ ]:


data5 = data


# In[23]:


data5["body"] = data5["body"].fillna(0)
data5["home.dest"] = data5["home.dest"].fillna("Desconocido")
data5.head(5)


# In[24]:


pd.isnull(data5["age"]).values.ravel().sum()


# In[25]:


data5["age"].fillna(data["age"].mean())


# In[26]:


data5["age"][1291]


# In[27]:


data5["age"].fillna(method="ffill")


# In[28]:


data5["age"].fillna(method="backfill")


# # Variables dummy

# In[29]:


data["sex"].head(10)


# In[ ]:


dummy_sex = pd.get_dummies(data["sex"], prefix="sex")


# In[31]:


dummy_sex.head(10)


# In[32]:


column_name=data.columns.values.tolist()
column_name


# In[ ]:


data = data.drop(["sex"], axis = 1)


# In[ ]:


data = pd.concat([data, dummy_sex], axis = 1)


# In[ ]:


def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy ], axis = 1)
    return df


# In[36]:


createDummies(data3, "sex")

