#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/Collab---v-3.8/notebooks/T2%20-%203%20-%20Data%20Cleaning%20-%20Agrupaci%C3%B3n%20de%20datos.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Agregación de datos por categoría

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


gender = ["Male", "Female"]
income = ["Poor", "Middle Class", "Rich"]


# In[ ]:


n = 500

gender_data = []
income_data = []

for i in range(0,500):
    gender_data.append(np.random.choice(gender))
    income_data.append(np.random.choice(income))


# In[4]:


gender_data[1:10]


# In[5]:


income_data[1:10]


# In[ ]:


#Z -> N(0,1)
#N(m, s) -> m + s * Z
height = 160 + 30 * np.random.randn(n)
weight = 65 + 25 * np.random.randn(n)
age = 30 + 12 * np.random.randn(n)
income = 18000 + 3500 * np.random.rand(n)


# In[ ]:


data = pd.DataFrame(
    {
        "Gender" : gender_data,
        "Economic Status" : income_data,
        "Height" : height,
        "Weight" : weight,
        "Age" : age,
        "Income" : income
    }
)


# In[8]:


data.head()


# ## Agrupación de datos

# In[ ]:


grouped_gender = data.groupby("Gender")


# In[10]:


grouped_gender.groups


# In[11]:


for names, groups in grouped_gender:
    print(names)
    print(groups)


# In[12]:


grouped_gender.get_group("Female")


# In[ ]:


double_group = data.groupby(["Gender", "Economic Status"])


# In[14]:


len(double_group)


# In[15]:


for names, groups in double_group:
    print(names)
    print(groups)


# ## Operaciones sobre datos agrupados

# In[16]:


double_group.sum()


# In[17]:


double_group.mean()


# In[18]:


double_group.size()


# In[19]:


double_group.describe()


# In[ ]:


grouped_income = double_group["Income"]


# In[21]:


grouped_income.describe()


# In[22]:


double_group.aggregate(
    {
        "Income": np.sum,
        "Age" : np.mean,
        "Height" : np.std
    }
)


# In[23]:


double_group.aggregate(
    {
        "Age" : np.mean,
        "Height" : lambda h:(np.mean(h))/np.std(h)
    }
)


# In[24]:


double_group.aggregate([np.sum, np.mean, np.std])


# In[25]:


double_group.aggregate([lambda x: np.mean(x) / np.std(x)])


# ## Filtrado de datos

# In[26]:


double_group["Age"].filter(lambda x: x.sum()>2400)


# ## Transformación de variables

# In[ ]:


zscore = lambda x : (x - x.mean())/x.std()


# In[ ]:


z_group = double_group.transform(zscore)


# In[ ]:


import matplotlib.pyplot as plt


# In[30]:


plt.hist(z_group["Age"])


# In[ ]:


fill_na_mean = lambda x : x.fillna(x.mean())


# In[32]:


double_group.transform(fill_na_mean)


# ## Operaciones diversas muy útiles

# In[33]:


double_group.head(1)


# In[34]:


double_group.tail(1)


# In[35]:


double_group.nth(32)


# In[36]:


double_group.nth(82)


# In[ ]:


data_sorted = data.sort_values(["Age", "Income"])


# In[38]:


data_sorted.head(10)


# In[ ]:


age_grouped = data_sorted.groupby("Gender")


# In[40]:


age_grouped.head(1)


# In[41]:


age_grouped.tail(1)


# # Conjunto de entrenamiento y conjunto de testing

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("/content/drive/My Drive/Curso Machine Learning con Python/datasets/customer-churn-model/Customer Churn Model.txt")


# In[ ]:


len(data)


# ## Dividir utilizando la distribución normal

# In[ ]:


a = np.random.randn(len(data))


# In[43]:


plt.hist(a)


# In[ ]:


check = (a<0.75) # No es el 75% de los datos, son los números que son < 0.75!!! 


# In[45]:


check


# In[47]:


plt.hist(check.astype(int))#Ha cambiado en la versión 3.7 de python y necesita hacer un cast de bool a entero


# In[ ]:


training = data[check]
testing = data[~check]


# In[49]:


len(training)


# In[50]:


len(testing)


# ## Con la libreria sklearn

# In[ ]:


from sklearn.model_selection import train_test_split# Ha cambiado en la 3.7 de Python


# In[ ]:


train, test = train_test_split(data, test_size = 0.2)


# In[54]:


len(train)


# In[55]:


len(test)


# ## Usando una función de shuffle

# In[ ]:


import numpy as np


# In[58]:


data.head()


# In[ ]:


import sklearn


# In[ ]:


data = sklearn.utils.shuffle(data)


# In[ ]:


cut_id = int(0.75*len(data))
train_data = data[:cut_id]
test_data = data[cut_id+1:]


# In[62]:


len(train_data)


# In[63]:


len(test_data)


# In[ ]:




