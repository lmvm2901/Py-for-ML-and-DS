#!/usr/bin/env python
# coding: utf-8

# # Modelos de Regresión Lineal
# ## Modelo con datos simulados
# * y = a + b * x
# * X : 100 valores distribuídos según una N(1.5, 2.5)
# * Ye = 5 + 1.9 * x + e
# * e estará distribuído según una N(0, 0.8)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


x = 1.5 + 2.5 * np.random.randn(100)


# In[3]:


res = 0 + 0.8 * np.random.randn(100)


# In[4]:


y_pred = 5 + 0.3 * x


# In[5]:


y_act = 5 + 0.3 * x + res


# In[6]:


x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()


# In[7]:


data = pd.DataFrame(
    {
        "x":x_list,
        "y_actual":y_act_list,
        "y_prediccion":y_pred_list
    }
)


# In[8]:


data.head()


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data["x"],data["y_prediccion"])
plt.plot(data["x"], data["y_actual"], "ro")
plt.plot(data["x"],y_mean, "g")
plt.title("Valor Actual vs Predicción")


# ## ¿Como es la predicción de buena?
# * SST = SSD + SSR
# * SST : Variabilidad de los datos con respecto de su media
# * SSD : Diferencia entre los datos originales y las predicciones que el modelo no es capaz de explicar (errores que deberían seguir una distribución normal)
# * SSR : Diferencia entre la regresión y el valor medio que el modelo busca explicar
# * R2 = SSR / SST, coeficiente de determinación entre 0 y 1

# In[12]:


y_m = np.mean(y_act)
data["SSR"]=(data["y_prediccion"]-y_m)**2
data["SSD"]=(data["y_prediccion"]-data["y_actual"])**2
data["SST"]=(data["y_actual"]-y_m)**2


# In[13]:


data.head()


# In[14]:


SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])


# In[15]:


SSR


# In[16]:


SSD


# In[17]:


SST


# In[18]:


SSR+SSD


# In[19]:


R2 = SSR/SST


# In[20]:


R2


# In[21]:


plt.hist(data["y_prediccion"]-data["y_actual"])


# ## Obteniendo la recta de regresión 
# 
# * y = a + b * x
# * b = sum((xi - x_m)*(y_i-y_m))/sum((xi-x_m)^2)
# * a = y_m - b * x_m

# In[22]:


x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])
x_mean, y_mean


# In[23]:


data["beta_n"] = (data["x"]-x_mean)*(data["y_actual"]-y_mean)
data["beta_d"] = (data["x"]-x_mean)**2


# In[24]:


beta = sum(data["beta_n"])/sum(data["beta_d"])


# In[25]:


alpha = y_mean - beta * x_mean


# In[26]:


alpha, beta


# El modelo lineal obtenido por regresión es:
# y = 5.042341442370516 + 1.9044490309709992 * x

# In[27]:


data["y_model"] = alpha + beta * data["x"]


# In[28]:


data.head()


# In[29]:


SSR = sum((data["y_model"]-y_mean)**2)
SSD = sum((data["y_model"]-data["y_actual"])**2)
SST = sum((data["y_actual"]-y_mean)**2)


# In[30]:


SSR, SSD, SST


# In[31]:


R2 = SSR / SST
R2


# In[32]:


y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data["x"],data["y_prediccion"])
plt.plot(data["x"], data["y_actual"], "ro")
plt.plot(data["x"],y_mean, "g")
plt.plot(data["x"], data["y_model"])
plt.title("Valor Actual vs Predicción")


# ## Error estándar de los residuos (RSE)

# In[33]:


RSE = np.sqrt(SSD/(len(data)-2))
RSE


# In[34]:


np.mean(data["y_actual"])


# In[35]:


RSE / np.mean(data["y_actual"])

