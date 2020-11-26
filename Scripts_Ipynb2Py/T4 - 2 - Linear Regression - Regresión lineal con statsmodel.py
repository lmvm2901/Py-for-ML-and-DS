#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T4%20-%202%20-%20Linear%20Regression%20-%20Regresión%20lineal%20con%20statsmodel-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Regresión lineal simple en Python
# ## El paquete statsmodel para regresión lineal

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("../datasets/ads/Advertising.csv")


# In[3]:


data.head()


# In[4]:


import statsmodels.formula.api as smf


# In[5]:


lm = smf.ols(formula="Sales~TV", data = data).fit()


# In[6]:


lm.params


# El modelo lineal predictivo sería 
# Sales = 7.032594 + 0.047537 * TV

# In[7]:


lm.pvalues


# In[8]:


lm.rsquared


# In[9]:


lm.rsquared_adj


# In[10]:


lm.summary()


# In[11]:


sales_pred = lm.predict(pd.DataFrame(data["TV"]))
sales_pred


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.plot(kind = "scatter", x = "TV", y ="Sales")
plt.plot(pd.DataFrame(data["TV"]), sales_pred, c="red", linewidth = 2)


# In[14]:


data["sales_pred"] = 7.032594 + 0.047537*data["TV"]


# In[15]:


data["RSE"] = (data["Sales"]-data["sales_pred"])**2


# In[16]:


SSD = sum(data["RSE"])
SSD


# In[17]:


RSE = np.sqrt(SSD/(len(data)-2))
RSE


# In[18]:


sales_m = np.mean(data["Sales"])


# In[19]:


sales_m


# In[20]:


error = RSE/sales_m


# In[21]:


error


# In[22]:


plt.hist((data["Sales"]-data["sales_pred"]))


# # Regresión lineal múltiple en Python
# ## El paquete statsmodel para regresión múltiple
# * Sales ~TV
# * Sales ~Newspaper
# * Sales ~Radio
# * Sales ~TV+Newspaper
# * Sales ~TV+Radio
# * Sales ~Newspaper+Radio
# * Sales ~TV+Newspaper+Radio

# In[23]:


#Añadir el Newspaper al modelo existente
lm2 = smf.ols(formula="Sales~TV+Newspaper", data = data).fit()


# In[24]:


lm2.params


# In[25]:


lm2.pvalues


# Sales = 5.774948+0.046901*TV + 0.044219*Newspaper

# In[26]:


lm2.rsquared


# In[27]:


lm2.rsquared_adj


# In[28]:


sales_pred = lm2.predict(data[["TV", "Newspaper"]])


# In[29]:


sales_pred


# In[30]:


SSD = sum((data["Sales"]-sales_pred)**2)


# In[31]:


SSD


# In[32]:


RSE = np.sqrt(SSD/(len(data)-2-1))


# In[33]:


RSE


# In[34]:


error = RSE / sales_m


# In[35]:


error


# In[36]:


lm2.summary()


# In[37]:


#Añadir la Radio al modelo existente
lm3 = smf.ols(formula="Sales~TV+Radio", data = data).fit()


# In[38]:


lm3.summary()


# In[39]:


sales_pred = lm3.predict(data[["TV", "Radio"]])
SSD = sum((data["Sales"]-sales_pred)**2)
RSE = np.sqrt(SSD/(len(data)-2-1))


# In[40]:


RSE


# In[41]:


RSE/sales_m


# In[42]:


#Añadir la Radio al modelo existente
lm4 = smf.ols(formula="Sales~TV+Radio+Newspaper", data = data).fit()


# In[43]:


lm4.summary()


# In[44]:


sales_pred = lm4.predict(data[["TV", "Radio","Newspaper"]])
SSD = sum((data["Sales"]-sales_pred)**2)
RSE = np.sqrt(SSD/(len(data)-3-1))


# In[45]:


RSE


# In[46]:


RSE/sales_m


# ## Multicolinealidad 
# #### Factor Inflación de la Varianza
# * VIF = 1 : Las variables no están correlacionadas
# * VIF < 5 : Las variables tienen una correlación moderada y se pueden quedar en el modelo
# * VIF >5 : Las variables están altamente correlacionadas y deben desaparecer del modelo.

# In[47]:


# Newspaper ~ TV + Radio -> R^2 VIF = 1/(1-R^2)
lm_n = smf.ols(formula="Newspaper~TV+Radio", data = data).fit()
rsquared_n = lm_n.rsquared
VIF = 1/(1-rsquared_n)
VIF


# In[48]:


# TV ~ Newspaper + Radio -> R^2 VIF = 1/(1-R^2)
lm_tv = smf.ols(formula="TV~Newspaper+Radio", data=data).fit()
rsquared_tv = lm_tv.rsquared
VIF = 1/(1-rsquared_tv)
VIF


# In[49]:


# Radio ~ TV + Newspaper -> R^2 VIF = 1/(1-R^2)
lm_r = smf.ols(formula="Radio~Newspaper+TV", data=data).fit()
rsquared_r = lm_r.rsquared
VIF = 1/(1-rsquared_r)
VIF


# In[50]:


lm3.summary()

