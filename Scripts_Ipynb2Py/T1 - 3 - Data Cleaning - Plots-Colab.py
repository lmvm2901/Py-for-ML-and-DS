#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T1%20-%203%20-%20Data%20Cleaning%20-%20Plots-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# # Plots y visualización de los datos

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/customer-churn-model/Customer Churn Model.txt")


# In[3]:


data


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#savefig("path_donde_guardar_im.jpeg")


# ### Scatter Plot

# In[5]:


data.plot(kind="scatter", x="Day Mins", y="Day Charge")


# In[6]:


data.plot(kind="scatter", x="Night Mins", y="Night Charge")


# In[7]:


figure, axs = plt.subplots(2,2, sharey=True, sharex=True)
data.plot(kind="scatter", x="Day Mins", y ="Day Charge", ax=axs[0][0])
data.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[0][1])
data.plot(kind="scatter", x="Day Calls", y ="Day Charge", ax=axs[1][0])
data.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[1][1])


# ### Histogramas de frecuencias

# In[8]:


k = int(np.ceil(1+np.log2(3333)))
plt.hist(data["Day Calls"], bins = k) #bins = [0,30,60,...,200]
plt.xlabel("Número de llamadas al día")
plt.ylabel("Frecuencia")
plt.title("Histograma del número de llamadas al día")


# ### Boxplot, diagrama de caja y bigotes

# In[9]:


plt.boxplot(data["Day Calls"])
plt.ylabel("Número de llamadas diarias")
plt.title("Boxplot de las llamadas diarias")


# In[10]:


data["Day Calls"].describe()


# In[11]:


IQR=data["Day Calls"].quantile(0.75)-data["Day Calls"].quantile(0.25)
IQR


# In[12]:


data["Day Calls"].quantile(0.25) - 1.5*IQR


# In[13]:


data["Day Calls"].quantile(0.75) + 1.5*IQR

