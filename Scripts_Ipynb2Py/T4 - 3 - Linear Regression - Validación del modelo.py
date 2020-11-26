#!/usr/bin/env python
# coding: utf-8

# # Dividir el dataset en conjunto de entrenamiento y de testing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("../datasets/ads/Advertising.csv")


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
Image(filename="resources/summary-lm.png")

