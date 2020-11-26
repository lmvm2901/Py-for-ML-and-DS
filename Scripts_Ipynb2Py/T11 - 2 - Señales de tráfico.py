#!/usr/bin/env python
# coding: utf-8

# # Reconocimiento de las señales de tráfico

# In[1]:


import tensorflow as tf
import os
import skimage.data as imd
import numpy as np


# In[2]:


def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory,d))]
    
    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                     for f in os.listdir(label_dir)
                     if f.endswith(".ppm")]
        
        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
        
    return images, labels


# In[3]:


main_dir = "../datasets/belgian/"
train_data_dir = os.path.join(main_dir, "Training")
test_data_dir = os.path.join(main_dir, "Testing")


# In[4]:


images, labels = load_ml_data(train_data_dir)


# In[5]:


images = np.array(images)


# In[6]:


labels = np.array(labels)


# In[7]:


images.ndim


# In[8]:


images.size


# In[9]:


images[0]


# In[10]:


labels.ndim


# In[11]:


labels.size


# In[12]:


len(set(labels))


# In[13]:


images.flags


# In[14]:


images.itemsize


# In[15]:


images.nbytes


# In[16]:


images.nbytes/images.itemsize


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.hist(labels, len(set(labels)))
plt.show()


# In[19]:


import random


# In[20]:


rand_signs = random.sample(range(0, len(labels)), 6)
rand_signs


# In[21]:


for i in range(len(rand_signs)):
    temp_im = images[rand_signs[i]]
    plt.subplot(1,6,i+1)
    plt.axis("off")
    plt.imshow(temp_im)
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))


# In[22]:


unique_labels = set(labels)
plt.figure(figsize=(16,16))
i = 1
for label in unique_labels:
    temp_im = images[list(labels).index(label)]
    plt.subplot(8,8, i)
    plt.axis("off")
    plt.title("Clase {0} ({1})".format(label, list(labels).count(label)))
    i +=1
    plt.imshow(temp_im)
plt.show()


# In[23]:


type(labels)


# # Modelo de Red Neuronal con TensorFlow
# * Las imágenes no todas son del mismo tamaño
# * Hay 62 clases de imágenes (desde la 0 hasta la 61)
# * La distribución de señales de tráfico no es uniforme (algunas salen más veces que otras)

# In[24]:


from skimage import transform


# In[25]:


w = 9999 
h = 9999
for image in images:
    if image.shape[0] < h:
        h = image.shape[0]
    if image.shape[1] < w:
        w = image.shape[1]
print("Tamaño mínimo: {0}x{1}".format(h,w))


# In[26]:


images30 = [transform.resize(image, (30,30)) for image in images]


# In[30]:


images30[0]


# In[27]:


rand_signs = random.sample(range(0, len(labels)), 6)
rand_signs
for i in range(len(rand_signs)):
    temp_im = images30[rand_signs[i]]
    plt.subplot(1,6,i+1)
    plt.axis("off")
    plt.imshow(temp_im)
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))


# In[28]:


from skimage.color import rgb2gray


# In[29]:


images30 = np.array(images30)
images30 = rgb2gray(images30)


# In[30]:


rand_signs = random.sample(range(0, len(labels)), 6)
rand_signs
for i in range(len(rand_signs)):
    temp_im = images30[rand_signs[i]]
    plt.subplot(1,6,i+1)
    plt.axis("off")
    plt.imshow(temp_im, cmap="gray")
    plt.subplots_adjust(wspace = 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))


# In[31]:


x = tf.placeholder(dtype = tf.float32, shape = [None, 30,30])
y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits=logits))

train_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

final_pred = tf.argmax(logits,1)

accuracy = tf.reduce_mean(tf.cast(final_pred, tf.float32))


# In[36]:


images_flat


# In[37]:


logits


# In[38]:


loss


# In[39]:


final_pred


# In[32]:


tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(601):
    
    _, accuracy_val = sess.run([train_opt, accuracy],
                              feed_dict= {
                                  x: images30,
                                  y: list(labels)
                              })
    #_, loss_val = sess.run([train_opt, loss],
    #                          feed_dict= {
    #                              x: images30,
    #                              y: list(labels)
    #                          })
    if i%50 == 0:
        print("EPOCH", i)
        print("Eficacia: ", accuracy_val)
        #print("Pérdidas:", loss_val)
    #print("Fin del Ecpoh ", i)


# # Evaluación de la red neuronal

# In[41]:


sample_idx = random.sample(range(len(images30)), 40)
sample_images = [images30[i] for i in sample_idx]
sample_labels = [labels[i] for i in sample_idx]


# In[42]:


prediction = sess.run([final_pred], feed_dict={x:sample_images})[0]


# In[43]:


prediction


# In[44]:


sample_labels


# In[45]:


plt.figure(figsize=(16,20))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(10,4,i+1)
    plt.axis("off")
    color = "green" if truth==predi else "red"
    plt.text(32,15, "Real:         {0}\nPrediccion:{1}".format(truth, predi),
            fontsize = 14, color = color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()


# In[46]:


test_images, test_labels = load_ml_data(test_data_dir)


# In[47]:


test_images30 = [transform.resize(im,(30,30)) for im in test_images]


# In[48]:


test_images30 = rgb2gray(np.array(test_images30))


# In[49]:


prediction = sess.run([final_pred], feed_dict={x:test_images30})[0]


# In[50]:


match_count = sum([int(l0 == lp) for l0, lp in zip(test_labels, prediction)])
match_count


# In[51]:


acc = match_count/len(test_labels)*100
print("Eficacia de la red neuronal: {:.2f}".format(acc))

