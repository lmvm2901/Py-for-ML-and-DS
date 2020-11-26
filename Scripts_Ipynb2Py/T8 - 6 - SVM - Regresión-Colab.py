#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T8%20-%206%20-%20SVM%20-%20Regresión-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # SVM para Regresión

# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[8]:


X = np.sort(5*np.random.rand(200,1),axis=0)
Y = np.sin(X).ravel()
Y[::5] += 3*(0.5 - np.random.rand(40))


# In[10]:


plt.scatter(X,Y, color="darkorange", label="data")


# In[11]:


from sklearn.svm import SVR


# In[12]:


C=1e3
svr_lin = SVR(kernel="linear", C=C)
svr_rbf = SVR(kernel="rbf", C=C, gamma=0.1)
svr_pol = SVR(kernel="poly", C=C, degree=3)


# In[13]:


y_lin = svr_lin.fit(X,Y).predict(X)
y_rbf = svr_rbf.fit(X,Y).predict(X)
y_pol = svr_pol.fit(X,Y).predict(X)


# In[14]:


lw = 2
plt.figure(figsize=(16,9))
plt.scatter(X,Y,color="darkorange", label ="data")
plt.plot(X,y_lin, color="navy", lw = lw, label = "SVM Lineal")
plt.plot(X,y_rbf, color="c", lw=lw, label="SVM Radial")
plt.plot(X,y_pol, color="cornflowerblue", label="SVM Polinómico")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Support Vector Regression")
plt.legend()
plt.show()


# In[ ]:




