#!/usr/bin/env python
# coding: utf-8

# # Creando nuestro propio KNN

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter


# In[2]:


dataset = {
    'k':[[1,2],[2,3],[3,1]],
    'r':[[6,5],[7,7],[8,6]]
}
new_point = [5,7]


# In[4]:


[[plt.scatter(ii[0],ii[1], s=50, color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_point[0],new_point[1], s = 100)


# In[33]:


def k_nearest_neighbors(data, predict, k = 3, verbose = False):
    
    if len(data) >= k:
        warnings.warn("K es un valor menor que el número total de elementos a votar!!")
    
    distances = []
    for group in data:
        for feature in data[group]:
            #d = sqrt((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2)
            #d = np.sqrt(np.sum((np.array(feature) - np.array(predict))**2))
            d = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([d, group])
    if verbose:
        print(distances)
    
    votes = [i[1] for i in sorted(distances)[:k]]#sorted ordena por la primera columna
    if verbose:
        print(votes)
    
    vote_result = Counter(votes).most_common(1)
    if verbose:
        print(vote_result)
    
    
    return vote_result[0][0]#[('r',2), ('k', 1)]


# In[15]:


new_point = [4,4.5]
result = k_nearest_neighbors(dataset, [new_point])
result


# In[16]:


[[plt.scatter(ii[0],ii[1], s=50, color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_point[0],new_point[1], s = 100, color=result)


# # Aplicando nuestro KNN al Dataset del Cancer

# In[17]:


import pandas as pd


# In[18]:


df = pd.read_csv("../datasets/cancer/breast-cancer-wisconsin.data.txt")


# In[19]:


df.replace("?", -99999, inplace=True)


# In[20]:


df.columns = ["name", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "class"]


# In[21]:


df.drop(["name"], 1, inplace=True)


# In[22]:


df.head()


# In[23]:


full_data = df.astype(float).values.tolist()


# In[24]:


full_data


# In[25]:


import random


# In[26]:


random.shuffle(full_data)


# In[27]:


test_size = 0.2


# In[28]:


train_set = {2:[],4:[]}
test_set = {2:[], 4:[]}


# In[29]:


train_data= full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]


# In[31]:


for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])


# In[34]:


correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct += 1
        total +=1
print("Eficacia del KNN = ",correct/total)


# In[ ]:




