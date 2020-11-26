#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T8%20-%201%20-%20SVM%20-%20Linear%20SVC-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Linear Support Vector Classifier

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


# In[16]:


X = [1,5,1.5,8,1,9]
Y = [2,8,1.8,8,0.6,11]


# In[4]:


plt.scatter(X,Y)
plt.show()


# In[18]:


data = np.array(list(zip(X,Y)))


# In[19]:


data


# In[20]:


target = [0, 1, 0, 1, 0, 1]


# In[21]:


classifier = svm.SVC(kernel="linear", C = 1.0)
classifier.fit(data, target)


# In[36]:


p = np.array([10.32, 12.67]).reshape(1,2)
print(p)
classifier.predict(p)


# * Modelo: w0 . x + w1 . y + e = 0
# * Ecuación del hiperplano en 2D: y = a . x + b 

# In[43]:


w = classifier.coef_[0]
w


# In[44]:


a = -w[0]/w[1]
a


# In[45]:


b = - classifier.intercept_[0]/w[1]
b


# In[46]:


xx = np.linspace(0,10)
yy = a * xx + b


# In[47]:


plt.plot(xx, yy, 'k-', label = "Hiperplano de separación")
plt.scatter(X, Y, c = target)
plt.legend()
plt.plot()


# In[ ]:




