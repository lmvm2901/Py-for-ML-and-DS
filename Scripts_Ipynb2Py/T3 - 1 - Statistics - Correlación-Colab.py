#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T3%20-%201%20-%20Statistics%20-%20Correlación-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# In[1]:


import pandas as pd


# In[21]:


data_ads = pd.read_csv("/content/python-ml-course/datasets/ads/Advertising.csv")


# In[3]:


data_ads.head()


# In[4]:


len(data_ads)


# In[5]:


import numpy as np


# In[6]:


data_ads["corrn"] = (data_ads["TV"] - np.mean(data_ads["TV"]))* (data_ads["Sales"] - np.mean(data_ads["Sales"]))


# In[7]:


data_ads.head()


# In[8]:


data_ads["corr1"] = (data_ads["TV"] - np.mean(data_ads["TV"]))**2


# In[9]:


data_ads.head()


# In[10]:


data_ads["corr2"] = (data_ads["Sales"] - np.mean(data_ads["Sales"]))**2


# In[11]:


data_ads.head()


# In[15]:


corrn = sum(data_ads["corrn"])/np.sqrt(sum(data_ads["corr1"]) * sum(data_ads["corr2"]))


# In[16]:


corrn


# In[17]:


def corr_coeff(df, var1, var2):
    df["corrn"] = (df[var1] - np.mean(df[var1]))* (df[var2] - np.mean(df[var2]))
    df["corr1"] = (df[var1] - np.mean(df[var1]))**2
    df["corr2"] = (df[var2] - np.mean(df[var2]))**2
    corr_p = sum(df["corrn"])/np.sqrt(sum(df["corr1"]) * sum(df["corr2"]))
    return corr_p


# In[18]:


corr_coeff(data_ads, "TV", "Sales")


# In[22]:


cols = data_ads.columns.values


# In[25]:


for x in cols:
    for y in cols:
        print(x + ", "+ y + " : " + str(corr_coeff(data_ads, x, y)))


# In[26]:


import matplotlib.pyplot as plt


# In[28]:


plt.plot(data_ads["TV"], data_ads["Sales"], "ro")
plt.title("Gasto en TV vs Ventas del Producto")


# In[33]:


plt.plot(data_ads["Radio"], data_ads["Sales"], "go")
plt.title("Gasto en Radio vs Ventas del Producto")


# In[34]:


plt.plot(data_ads["Newspaper"], data_ads["Sales"], "bo")
plt.title("Gasto en Periódico vs Ventas del Producto")


# In[39]:


data_ads = pd.read_csv("/content/python-ml-course/datasets/ads/Advertising.csv")
data_ads.corr()


# In[40]:


plt.matshow(data_ads.corr())

