#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T2%20-%201%20-%20Data%20Cleaning%20-%20Data%20Wrangling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Data Wrangling - La cirugía de los datos

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# El **data wrangling**, a veces denominada **data munging**, es el proceso de transformar y mapear datos de un dataset *raw* (en bruto) en otro formato con la intención de hacerlo más apropiado y valioso para una variedad de propósitos posteriores, como el análisis. Un **data wrangler** es una persona que realiza estas operaciones de transformación.
# 
# Esto puede incluir munging, visualización de datos, agregación de datos, entrenamiento de un modelo estadístico, así como muchos otros usos potenciales. La oscilación de datos como proceso generalmente sigue un conjunto de pasos generales que comienzan extrayendo los datos en forma cruda del origen de datos, dividiendo los datos en bruto usando algoritmos (por ejemplo, clasificación) o analizando los datos en estructuras de datos predefinidas, y finalmente depositando el contenido resultante en un sistema de almacenamiento (o silo) para su uso futuro.

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/customer-churn-model/Customer Churn Model.txt")


# In[5]:


data.head()


# ### Crear un subconjunto de datos

# #### Subconjunto de columna o columnas

# In[ ]:


account_length = data["Account Length"]


# In[7]:


account_length.head()


# In[8]:


type(account_length)


# In[ ]:


subset = data[["Account Length", "Phone", "Eve Charge", "Day Calls"]]


# In[10]:


subset.head()


# In[11]:


type(subset)


# In[12]:


desired_columns = ["Account Length", "Phone", "Eve Charge", "Night Calls"]
subset = data[desired_columns]
subset.head()


# In[13]:


desired_columns = ["Account Length", "VMail Message", "Day Calls"]
desired_columns


# In[14]:


all_columns_list = data.columns.values.tolist()
all_columns_list


# In[15]:


sublist = [x for x in all_columns_list if x not in desired_columns]
sublist


# In[16]:


subset = data[sublist]
subset.head()


# #### Subconjunto de filas

# In[17]:


data[1:25]


# In[18]:


data[10:35]


# In[19]:


data[:8] # CORRECCIÓN: es lo mismo que data[0:8]


# In[20]:


data[3320:]


# #### Subconjuntos de filas con condiciones booleanas

# In[21]:


##Usuarios con Day Mins > 300
data1 = data[data["Day Mins"]>300]
data1.shape


# In[22]:


##Usuarios de Nueva York (State = "NY")
data2 = data[data["State"]=="NY"]
data2.shape


# In[23]:


##AND -> &
data3 = data[(data["Day Mins"]>300) & (data["State"]=="NY")]
data3.shape


# In[24]:


##OR -> |
data4 = data[(data["Day Mins"]>300) | (data["State"]=="NY")]
data4.shape


# In[25]:


data5 = data[data["Day Calls"]<data["Night Calls"]]
data5.shape


# In[26]:


data6 = data[data["Day Mins"]<data["Night Mins"]]
data6.shape


# In[27]:


##Minutos de día, de noche y Longitud de la Cuenta de los primeros 50 individuos
subset_first_50 = data[["Day Mins", "Night Mins", "Account Length"]][:50]
subset_first_50.head()


# In[28]:


subset[:10]


# #### Filtrado con ix -> loc e iloc

# In[ ]:


data.ix[1:10, 3:6] ## Primeras 10 filas, columnas de la 3 a la 6


# In[29]:


data.iloc[1:10, 3:6]


# In[30]:


data.iloc[:,3:6] ##Todas las filas para las columnas entre la 3 y la 6
data.iloc[1:10,:] ##Todas las columnas para las filas de la 1 a la 10


# In[31]:


data.iloc[1:10, [2,5,7]]


# In[32]:


data.iloc[[1,5,8,36], [2,5,7]]


# In[33]:


data.loc[[1,5,8,36], ["Area Code", "VMail Plan", "Day Mins"]]


# #### Insertar nuevas filas en el dataframe

# In[ ]:


data["Total Mins"] = data["Day Mins"] + data["Night Mins"] + data["Eve Mins"]


# In[35]:


data["Total Mins"].head()


# In[ ]:


data["Total Calls"] = data["Day Calls"] + data["Night Calls"] + data["Eve Calls"]


# In[37]:


data["Total Calls"].head()


# In[38]:


data.shape


# In[39]:


data.head()


# ### Generación aleatoria de números

# In[ ]:


import numpy as np


# In[41]:


##Generar un número aleatorio entero entre 1 y 100
np.random.randint(1,100)


# In[42]:


##La forma más clásica de generar un número aleatorio es entre 0 y 1 (con decimales)
np.random.random()


# In[ ]:


##Función que genera una lista de n números aleatorios enteros dentro del intervalo [a,b]
def randint_list(n, a, b):
    x = []
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x


# In[44]:


randint_list(25, 1, 50)


# In[ ]:


import random


# In[46]:


for i in range(10):
    print(random.randrange(0, 100,7))


# #### Shuffling

# In[47]:


a = np.arange(100)
a


# In[ ]:


np.random.shuffle(a)


# In[49]:


a


# #### Choice

# In[50]:


data.head()


# In[51]:


data.shape


# In[52]:


column_list = data.columns.values.tolist()
column_list


# In[53]:


np.random.choice(column_list)


# #### Seed

# In[54]:


np.random.seed(2018)
for i in range(5):
    print(np.random.random())


# In[ ]:




