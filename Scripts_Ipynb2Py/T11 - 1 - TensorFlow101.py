#!/usr/bin/env python
# coding: utf-8

# # Introducción a Tensor Flow

# In[17]:


import tensorflow as tf


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




