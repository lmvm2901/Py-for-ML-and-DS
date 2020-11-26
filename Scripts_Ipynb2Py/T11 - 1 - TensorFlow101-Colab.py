#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T11%20-%201%20-%20TensorFlow101-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '1.x')


# # Introducción a Tensor Flow

# In[17]:


import tensorflow as tf
print(tensorflow.__version__)


# In[18]:


x1 = tf.constant([1,2,3,4,5])
x2 = tf.constant([6,7,8,9,10])


# In[19]:


res = tf.multiply(x1,x2)
print(res)


# In[20]:


sess = tf.Session()
print(sess.run(res))
sess.close()


# In[21]:


with tf.Session() as sess:
    output = sess.run(res)
    print(output)


# In[22]:


config = tf.ConfigProto(log_device_placement = True)
config = tf.ConfigProto(allow_soft_placement = True)


# In[ ]:




