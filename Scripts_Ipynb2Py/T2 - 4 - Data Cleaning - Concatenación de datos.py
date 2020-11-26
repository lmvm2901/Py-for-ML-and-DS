#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T2%20-%204%20-%20Data%20Cleaning%20-%20Concatenaci%C3%B3n%20de%20datos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Concatenar y apendizar data sets

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# ## El ejemplo del vino blanco y el vino tinto

# Data Set Information:
# 
# These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
# 
# I think that the initial data set had around 30 variables, but for some reason I only have the 13 dimensional version. I had a list of what the 30 or so variables were, but a.) I lost it, and b.), I would not know which 13 variables are included in the set. 
# 
# The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it ) 
# 
# 1. Alcohol 
# 2. Malic acid 
# 3. Ash 
# 4. Alcalinity of ash 
# 5. Magnesium 
# 6. Total phenols 
# 7. Flavanoids 
# 8. Nonflavanoid phenols 
# 9. Proanthocyanins 
# 10. Color intensity 
# 11. Hue 
# 12. OD280/OD315 of diluted wines 
# 13. Proline 
# 
# In a classification context, this is a well posed problem with "well behaved" class structures. A good data set for first testing of a new classifier, but not very challenging.
# 
# 
# Attribute Information:
# 
# All attributes are continuous 
# 
# No statistics available, but suggest to standardise variables for certain uses (e.g. for us with classifiers which are NOT scale invariant) 
# 
# NOTE: 1st attribute is class identifier (1-3)

# In[ ]:


import pandas as pd


# In[6]:


red_wine = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/wine/winequality-red.csv", sep=";")
red_wine.head()


# In[7]:


red_wine.columns.values


# In[8]:


red_wine.shape


# In[9]:


white_wine = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/wine/winequality-white.csv", sep = ";")
white_wine.head()


# In[10]:


white_wine.columns.values


# In[11]:


white_wine.shape


# En python, tenemos dos tipos de ejes, 
# * axis = 0 denota el eje horizontal
# * axis = 1 denota el eje vertical

# In[ ]:


wine_data = pd.concat([red_wine, white_wine], axis = 0)


# In[13]:


wine_data.shape


# In[14]:


wine_data.head()


# In[ ]:


data1 = wine_data.head(10)
data2 = wine_data[300:310]
data3 = wine_data.tail(10)


# In[ ]:


wine_scramble = pd.concat([data1, data2, data3], axis = 0)


# In[17]:


wine_scramble


# In[18]:


wine_scramble = pd.concat([data2, data1, data3], axis = 0)
wine_scramble


# ## Datos distribuidos 

# In[21]:


import pandas as pd
data = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/distributed-data/001.csv")
data.head()


# In[22]:


data.shape


# * Importar el primer fichero
# * Hacemos un bucle para ir recorriendo todos y cada uno de los ficheros. 
#     * Importante tener una consistencia en el nombre de los ficheros 
#     * Importamos los ficheros uno a uno
#     * Cada uno de ellos debe apendizarse (añadirse al final) del primer fichero que ya habíamos cargado
# * Repetimos el bucle hasta que no queden ficheros

# In[ ]:


filepath = "/content/drive/My Drive/Curso Machine Learning con Python/datasets/distributed-data/"

data = pd.read_csv(filepath+"001.csv")
final_length = len(data)

for i in range(2,333):
    if i < 10:
        filename = "00" + str(i)
    if 10 <= i < 100:
        filename = "0" + str(i)
    if i >= 100:
        filename = str(i)
    file = filepath + filename + ".csv"
    
    temp_data = pd.read_csv(file)
    final_length += len(temp_data)
    
    data = pd.concat([data, temp_data], axis = 0)


# In[28]:


data.shape


# In[29]:


data.tail()


# In[30]:


data.head()


# In[31]:


final_length == data.shape[0]


# # Joins de datasets

# In[ ]:


filepath = "/content/drive/My Drive/Curso Machine Learning con Python/datasets/athletes/"


# In[ ]:


data_main = pd.read_csv(filepath + "Medals.csv", encoding= "ISO-8859-1")


# In[ ]:


data_main.head()


# In[ ]:


a = data_main["Athlete"].unique().tolist()
len(a)


# In[ ]:


data_main.shape


# In[ ]:


data_country = pd.read_csv(filepath + "Athelete_Country_Map.csv", encoding = "ISO-8859-1")


# In[ ]:


data_country.head()


# In[ ]:


len(data_country)


# In[ ]:


data_country[data_country["Athlete"] == "Aleksandar Ciric"]


# In[ ]:


data_sports = pd.read_csv(filepath + "Athelete_Sports_Map.csv", encoding="ISO-8859-1")


# In[36]:


data_sports.head()


# In[ ]:


len(data_sports)


# In[ ]:


data_sports[(data_sports["Athlete"]=="Chen Jing") | 
            (data_sports["Athlete"]=="Richard Thompson") | 
            (data_sports["Athlete"]=="Matt Ryan")
           ]


# In[ ]:


data_country_dp = data_country.drop_duplicates(subset="Athlete")


# In[ ]:


len(data_country_dp)==len(a)


# In[ ]:


data_main_country = pd.merge(left = data_main, right = data_country_dp,
                            left_on="Athlete", right_on = "Athlete")


# In[ ]:


data_main_country.head()


# In[ ]:


data_main_country.shape


# In[ ]:


data_main_country[data_main_country["Athlete"] == "Aleksandar Ciric"]


# In[ ]:


data_sports_dp = data_sports.drop_duplicates(subset="Athlete")


# In[ ]:


len(data_sports_dp)==len(a)


# In[ ]:


data_final = pd.merge(left=data_main_country, right=data_sports_dp,
                     left_on="Athlete", right_on="Athlete")


# In[ ]:


data_final.head()


# In[ ]:


data_final.shape


# ## Tipos de Joins

# In[ ]:


from IPython.display import Image
import numpy as np


# **Inner Join <= A (Left Join), B (Right Join) <= Outer Join**

# In[ ]:


out_athletes = np.random.choice(data_main["Athlete"], size = 6, replace = False)


# In[ ]:


out_athletes


# In[ ]:


data_country_dlt = data_country_dp[(~data_country_dp["Athlete"].isin(out_athletes)) & 
                                   (data_country_dp["Athlete"] != "Michael Phelps")]

data_sports_dlt = data_sports_dp[(~data_sports_dp["Athlete"].isin(out_athletes)) &
                                (data_sports_dp["Athlete"] != "Michael Phelps")]

data_main_dlt = data_main[(~data_main["Athlete"].isin(out_athletes)) & 
                         (data_main["Athlete"] != "Michael Phelps")]


# In[ ]:


len(data_country_dlt)


# In[ ]:


len(data_sports_dlt)


# In[ ]:


len(data_main_dlt)


# ## Inner Join
# * Devuelve un data frame con las filas que tienen valor tanto en el primero como en el segundo data frame que estamos uniendo
# * El número de filas será igual al número de filas **comunes** que tengas ambos data sets
#     * Data Set A tiene 60 filas
#     * Data Set B tiene 50 filas
#     * Ambos comparten 30 filas
#     * Entonces A Inner Join B tendrá 30 filas
# * En términos de teoría de conjuntos, se trata de la intersección de los dos conjuntos

# In[ ]:


Image(filename="resources/inner-join.png")


# In[ ]:


# data_main contiene toda la info
# data_country_dlt le falta la info de 7 atletas
merged_inner = pd.merge(left = data_main, right = data_country_dlt,
                       how = "inner", left_on = "Athlete", right_on = "Athlete")


# In[ ]:


len(merged_inner)


# In[ ]:


merged_inner.head()


# ## Left Join
# * Devuelve un data frame con las filas que tuvieran valor en el dataset de la izquierda, sin importar si tienen correspondencia en el de la derecha o no.
# * Las filas del data frame final que no correspondan a ninguna fila del data frame derecho, tendrán NAs en las columnas del data frame derecho.
# * El número de filas será igual al número de filas del data frame izquierdo
#     * Data Set A tiene 60 filas
#     * Data Set B tiene 50 filas
#     * Entonces A Left Join B tendrá 60 filas
# * En términos de teoría de conjuntos, se trata del propio data set de la izquierda quien, además tiene la intersección en su interior.

# In[ ]:


Image(filename="resources/left-join.png")


# In[ ]:


merged_left = pd.merge(left = data_main, right = data_country_dlt, 
                      how = "left", left_on = "Athlete", right_on = "Athlete")
len(merged_left)


# In[ ]:


merged_left.head()


# ## Right Join
# * Devuelve un data frame con las filas que tuvieran valor en el dataset de la derecha, sin importar si tienen correspondencia en el de la izquierda o no.
# * Las filas del data frame final que no correspondan a ninguna fila del data frame izquierdo, tendrán NAs en las columnas del data frame izquierdo.
# * El número de filas será igual al número de filas del data frame derecho
#     * Data Set A tiene 60 filas
#     * Data Set B tiene 50 filas
#     * Entonces A Right Join B tendrá 50 filas
# * En términos de teoría de conjuntos, se trata del propio data set de la derecha quien, además tiene la intersección en su interior.

# In[ ]:


Image(filename="resources/right-join.png")


# In[ ]:


merged_right = pd.merge(left = data_main_dlt, right = data_country_dp,
                       how = "right", left_on = "Athlete", right_on = "Athlete")
len(merged_right)


# In[ ]:


merged_right.tail(10)


# ## Outer Join
# * Devuelve un data frame con todas las filas de ambos, reemplazando las ausencias de uno o de otro con NAs en la región específica..
# * Las filas del data frame final que no correspondan a ninguna fila del data frame derecho (o izquierdo), tendrán NAs en las columnas del data frame derecho (o izquierdo).
# * El número de filas será igual al máximo número de filas de ambos data frames
#     * Data Set A tiene 60 filas
#     * Data Set B tiene 50 filas
#     * Ambos comparten 30 filas
#     * Entonces A Outer Join B tendrá 60 + 50 - 30 = 80 filas
# * En términos de teoría de conjuntos, se trata de la unión de conjuntos.

# In[ ]:


Image(filename="resources/outer-join.png")


# In[ ]:


data_country_jb = data_country_dlt.append(
    {
        "Athlete": "Juan Gabriel Gomila",
        "Country": "España"
    },ignore_index = True
)


# In[ ]:


merged_outer = pd.merge(left = data_main, right=data_country_jb,
                       how = "outer", left_on = "Athlete", right_on="Athlete")
len(merged_outer)


# In[ ]:


merged_outer.head()


# In[ ]:


merged_outer.tail()


# In[ ]:


len(data_main)


# In[ ]:


len(data_main_dlt)


# In[ ]:


len(data_country_dp)


# In[ ]:


len(data_country_dlt)


# In[ ]:


len(merged_inner)


# In[ ]:


len(merged_left)


# In[ ]:


len(merged_right)


# In[ ]:


len(merged_outer)

