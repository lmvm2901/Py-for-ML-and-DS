#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T4%20-%203%20-%20Linear%20Regression%20-%20Validación%20del%20modelo-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Dividir el dataset en conjunto de entrenamiento y de testing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("/content/python-ml-course/datasets/ads/Advertising.csv")


# In[3]:


a = np.random.randn(len(data))


# In[4]:


plt.hist(a)


# In[5]:


check = (a<0.8)
training = data[check]
testing = data[~check]


# In[6]:


len(training), len(testing)


# In[7]:


import statsmodels.formula.api as smf
lm = smf.ols(formula="Sales~TV+Radio", data=training).fit()


# In[8]:


lm.summary()


# Sales = 2.9336 + 0.0465 * TV + 0.1807 * Radio

# ## Validación del modelo con el conjunto de testing

# In[9]:


sales_pred = lm.predict(testing)
sales_pred


# In[10]:


SSD = sum((testing["Sales"]-sales_pred)**2)
SSD


# In[11]:


RSE = np.sqrt(SSD/(len(testing)-2-1))
RSE


# In[12]:


sales_mean = np.mean(testing["Sales"])
error = RSE/sales_mean
error


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.plot(kind = "scatter", x = "TV", y ="Sales")
#plt.plot(pd.DataFrame(data["TV"]), sales_pred, c="red", linewidth = 2)


# In[14]:


from IPython.display import Image
Image(filename="/content/python-ml-course/notebooks/resources/summary-lm.png")

