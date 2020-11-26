#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T5%20-%202%20-%20Logistic%20Regression%20-%20Implementación-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Clonamos el repositorio para obtener los dataSet

# In[ ]:


get_ipython().system('git clone https://github.com/joanby/python-ml-course.git')


# # Damos acceso a nuestro Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
# Test it
get_ipython().system("ls '/content/drive/My Drive' ")


# # Implementación el método de la máxima verosimilitud para la regresión logística

# ### Definir la función de entorno L(b)

# In[1]:


from IPython.display import display, Math, Latex
display(Math(r'L(\beta)=\sum_{i=1}^n P_i^{y_i}(1-Pi)^{y_i}'))


# In[2]:


def likelihood(y, pi):
    import numpy as np
    total_sum = 1
    sum_in = list(range(1, len(y)+1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i]==1, pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum


# ### Calcular las probabilidades para cada observación

# In[3]:


display(Math(r'P_i = P(x_i) = \frac{1}{1+e^{-\sum_{j=0}^k\beta_j\cdot x_{ij}}} '))


# In[4]:


def logitprobs(X,beta):
    import numpy as np
    n_rows = np.shape(X)[0]
    n_cols = np.shape(X)[1]
    pi=list(range(1,n_rows+1))
    expon=list(range(1,n_rows+1))
    for i in range(n_rows):
        expon[i] = 0
        for j in range(n_cols):
            ex=X[i][j] * beta[j]
            expon[i] = ex + expon[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            pi[i]=1/(1+np.exp(-expon[i]))
    return pi


# ### Calcular la matriz diagonal W

# In[5]:


display(Math(r'W= diag(P_i \cdot (1-P_i))_{i=1}^n'))


# In[6]:


def findW(pi):
    import numpy as np
    n = len(pi)
    W = np.zeros(n*n).reshape(n,n)
    for i in range(n):
        print(i)
        W[i,i]=pi[i]*(1-pi[i])
        W[i,i].astype(float)
    return W


# ### Obtener la solución de la función logística

# In[7]:


display(Math(r"\beta_{n+1} = \beta_n -\frac{f(\beta_n)}{f'(\beta_n)}"))
display(Math(r"f(\beta) = X(Y-P)"))
display(Math(r"f'(\beta) = XWX^T"))


# In[8]:


def logistics(X, Y, limit):
    import numpy as np
    from numpy import linalg
    nrow = np.shape(X)[0]
    bias = np.ones(nrow).reshape(nrow,1)
    X_new = np.append(X, bias, axis = 1)
    ncol = np.shape(X_new)[1]
    beta = np.zeros(ncol).reshape(ncol,1)
    root_dif = np.array(range(1,ncol+1)).reshape(ncol,1)
    iter_i = 10000
    while(iter_i>limit):
        print("Iter:i"+str(iter_i) + ", limit:" + str(limit))
        pi = logitprobs(X_new, beta)
        print("Pi:"+str(pi))
        W = findW(pi)
        print("W:"+str(W))
        num = (np.transpose(np.matrix(X_new))*np.matrix(Y - np.transpose(pi)).transpose())
        den = (np.matrix(np.transpose(X_new))*np.matrix(W)*np.matrix(X_new))
        root_dif = np.array(linalg.inv(den)*num)
        beta = beta + root_dif
        print("Beta: "+str(beta))
        iter_i = np.sum(root_dif*root_dif)
        ll = likelihood(Y, pi)
    return beta


# ## Comprobación experimental

# In[9]:


import numpy as np


# In[10]:


X = np.array(range(10)).reshape(10,1)


# In[11]:


X


# In[12]:


Y = [0,0,0,0,1,0,1,0,1,1]


# In[13]:


bias = np.ones(10).reshape(10,1)
X_new = np.append(X,bias,axis=1)


# In[14]:


X_new


# In[15]:


a = logistics(X,Y,0.00001)


# In[16]:


ll = likelihood(Y, logitprobs(X,a))


# In[17]:


ll


# In[18]:


Y = 0.66220827 * X -3.69557172


# # Con el paquete statsmodel de python

# In[19]:


import statsmodels.api as sm
import pandas as pd
from pandas import Timestamp


# In[20]:


Y = (Y - np.min(Y))/np.ptp(Y)
logit_model = sm.Logit(Y,X_new)


# In[21]:


result = logit_model.fit()


# In[22]:


print(result.summary2())

