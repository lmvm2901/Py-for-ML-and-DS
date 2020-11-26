#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T2%20-%202%20-%20Data%20Cleaning%20-%20Funciones%20de%20distribuci%C3%B3n%20de%20probabilidad.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Funciones de distribución de probabilidades
# ## Distribución Uniforme

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


a = 1
b = 100
n = 1000000
data = np.random.uniform(a, b, n)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data)


# ## Distribución Normal

# In[ ]:


data = np.random.randn(1000000)


# In[6]:


x = range(1,1000001)
plt.plot(x, data)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(data)


# In[8]:


plt.plot(x,sorted(data))


# In[9]:


mu = 5.5
sd = 2.5
Z_10000 = np.random.randn(10000)
data = mu + sd * Z_10000 # Z = (X - mu) / sd -> N(0,1), X = mu + sd * Z
plt.hist(data)


# In[10]:


data = np.random.randn(2,4)
data


# ## La simulación de Monte Carlo

# * Generamos dos números aleatorios uniforme x e y entre 0 y 1 en total 1000 veces.
# * Calcularemos $z = x^2 + y^2$:
#     * Si $z < 1 \rightarrow$ estamos dentro del círculo.
#     * Si $z \geq 1 \rightarrow$ estamos fuera del círculo.
# * Calculamos el número total de veces que están dentro del círculo y lo dividimos entre el número total de intentos para obtener una aproximación de la probabilidad de caer dentro del círculo.
# * Usamos dicha probabilidad para aproximar el valor de π.
# * Repetimos el experimento un número suficiente de veces (por ejemplo 100), para obtener (100) diferentes aproximaciones de π. 
# * Calculamos el promedio de los 100 experimentos anteriores para dar un valor final de π.
#     

# In[ ]:


def pi_montecarlo(n, n_exp):
    pi_avg = 0
    pi_value_list = []
    for i in range(n_exp):
        value = 0
        x = np.random.uniform(0,1,n).tolist()
        y = np.random.uniform(0,1,n).tolist()
        for j in range(n):
            z = np.sqrt(x[j] * x[j] + y[j] * y[j])
            if z<=1:
                value += 1
        float_value = float(value)
        pi_value = float_value * 4 / n
        pi_value_list.append(pi_value)
        pi_avg += pi_value

    pi = pi_avg/n_exp

    print(pi)
    fig = plt.plot(pi_value_list)
    return (pi, fig)


# In[12]:


pi_montecarlo(10000, 200)


# ### Dummy Data Sets

# In[ ]:


n = 1000000
data = pd.DataFrame(
    {
        'A' : np.random.randn(n),
        'B' : 1.5 + 2.5 * np.random.randn(n),
        'C' : np.random.uniform(5, 32, n)
    }
)


# In[14]:


data.describe()


# In[15]:


plt.hist(data["A"])


# In[16]:


plt.hist(data["B"])


# In[17]:


plt.hist(data["C"])


# In[ ]:


data = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/customer-churn-model/Customer Churn Model.txt")


# In[19]:


data.head()


# In[ ]:


colum_names = data.columns.values.tolist()


# In[21]:


a = len(colum_names)
a


# In[ ]:


new_data = pd.DataFrame(
    {
        'Column Name': colum_names,
        'A' : np.random.randn(a),
        'B' : np.random.uniform(0,1,a)
    }, index = range(42, 42 + a)
)


# In[23]:


new_data


# In[ ]:




