#!/usr/bin/env python
# coding: utf-8

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




