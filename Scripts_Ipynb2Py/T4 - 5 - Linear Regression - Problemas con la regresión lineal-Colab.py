#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T4%20-%205%20-%20Linear%20Regression%20-%20Problemas%20con%20la%20regresión%20lineal-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # El tratamiento de las variables categóricas

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv("/content/python-ml-course/datasets/ecom-expense/Ecom Expense.csv")


# In[3]:


df.head()


# In[4]:


dummy_gender = pd.get_dummies(df["Gender"], prefix = "Gender")
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix = "City")


# In[5]:


dummy_gender.head()


# In[6]:


dummy_city_tier.head()


# In[7]:


column_names = df.columns.values.tolist()
column_names


# In[8]:


df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new.head()


# In[9]:


df_new = df_new[column_names].join(dummy_city_tier)
df_new.head()


# In[10]:


feature_cols = ["Monthly Income", "Transaction Time", 
                "Gender_Female", "Gender_Male", 
                "City_Tier 1", "City_Tier 2", "City_Tier 3",
                "Record"]


# In[11]:


X = df_new[feature_cols]
Y = df_new["Total Spend"]


# In[12]:


lm = LinearRegression()
lm.fit(X,Y)


# In[13]:


print(lm.intercept_)
print(lm.coef_)


# In[14]:


list(zip(feature_cols, lm.coef_))


# In[15]:


lm.score(X,Y)


# El modelo puede ser escrito como:
# * Total_Spend = -79.41713030137362 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545+'Gender_Female'* -131.02501325554567 + 'Gender_Male'* 131.0250132555456+'City_Tier 1'* 76.76432601049527 + 'City_Tier 2'* 55.138974309232474 + 'City_Tier 3'* -131.9033003197278+'Record'* 772.2334457445648
#     * Si es hombre y vive en CT1: Total_Spend = 128.37220896466724 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545+'Record'* 772.2334457445648
#     * Si es hombre y vive en CT2: Total_Spend = 106.74685726340445 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545 +'Record'* 772.2334457445648
#     * Si es hombre y vive en CT3: Total_Spend = -80.29541736555583 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545+'Record'* 772.2334457445648
#     * Si es mujer y vive en CT1: Total_Spend = -79.41713030137362 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545 - 131.0250132555456+ 76.76432601049527 +'Record'* 772.2334457445648
#     * Si es mujer y vive en CT2: Total_Spend = -79.41713030137362 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545 - 131.0250132555456+ 55.138974309232474  +'Record'* 772.2334457445648
#     * Si es mujer y vive en CT3: Total_Spend = -79.41713030137362 + 'Monthly Income'* 0.14753898049205738 + 'Transaction Time'* 0.15494612549589545 - 131.0250132555456-131.9033003197278 +'Record'* 772.2334457445648

# In[16]:


-79.41713030137362 + 131.0250132555456-131.9033003197278


# In[17]:


df_new["prediction"] = -79.41713030137362 + df_new['Monthly Income']*0.14753898049205738 + df_new['Transaction Time']* 0.15494612549589545+ df_new['Gender_Female'] * (-131.02501325554567) + df_new['Gender_Male'] * 131.0250132555456+ df_new['City_Tier 1']* 76.76432601049527 +  df_new['City_Tier 2']* 55.138974309232474 + df_new['City_Tier 3']* (-131.9033003197278)+ df_new['Record']* 772.2334457445648


# In[18]:


df_new.head()


# In[19]:


SSD = np.sum((df_new["prediction"] - df_new["Total Spend"])**2)


# In[20]:


SSD


# In[21]:


RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1))


# In[22]:


RSE


# In[23]:


sales_mean=np.mean(df_new["Total Spend"])


# In[24]:


sales_mean


# In[25]:


error = RSE/sales_mean


# In[26]:


error*100


# ## Eliminar variables dummy redundantes

# In[27]:


dummy_gender = pd.get_dummies(df["Gender"], prefix="Gender").iloc[:,1:]
dummy_gender.head()


# In[28]:


dummy_city_tier = pd.get_dummies(df["City Tier"], prefix="City").iloc[:,1:]
dummy_city_tier.head()


# In[29]:


column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new = df_new[column_names].join(dummy_city_tier)
df_new.head()


# In[30]:


feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male", "City_Tier 2", "City_Tier 3", "Record"]
X = df_new[feature_cols]
Y = df_new["Total Spend"]
lm = LinearRegression()
lm.fit(X,Y)


# In[31]:


print(lm.intercept_)


# In[32]:


list(zip(feature_cols, lm.coef_))


# In[33]:


lm.score(X,Y)


# Coeficientes con todas las variables en el modelo
# * ('Monthly Income', 0.14753898049205738),
# * ('Transaction Time', 0.15494612549589545),
# * ('Gender_Female', -131.02501325554567),
# * ('Gender_Male', 131.0250132555456),
# * ('City_Tier 1', 76.76432601049527),
# * ('City_Tier 2', 55.138974309232474),
# * ('City_Tier 3', -131.9033003197278),
# * ('Record', 772.2334457445648)
#  
#  Coeficientes tras enmascarar las variables dummy pertinentes
# * 'Monthly Income', 0.14753898049205744),
# * ('Transaction Time', 0.15494612549589631),
# * ('Gender_Male', 262.05002651109595),
# * ('City_Tier 2', -21.62535170126296),
# * ('City_Tier 3', -208.66762633022324),
# * ('Record', 772.2334457445635)]
# 
# Los cambios se reflejan en
# * Gender_Male: 
#     * antes -> 131.02, 
#     * después -> 262.05 = ( 131.02 - (-131.02))
# * Gender_Female: 
#     * antes -> -131.02,
#     * después -> 0
# * CT1: 
#     * antes -> 76.76,
#     * después -> 0
# * CT2: 
#     * antes -> 55.13, 
#     * después -> -21.62 = (55.13 - 76.76)
# * CT3: 
#     * antes -> -131.90, 
#     * después -> -208.66 = (-131.90 - 76.76)

# # Transformación de variables para conseguir una relación no lineal

# In[34]:


import pandas as pd


# In[35]:


data_auto = pd.read_csv("/content/python-ml-course/datasets/auto/auto-mpg.csv")
data_auto.head()


# In[36]:


data_auto.shape


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
data_auto["mpg"] = data_auto["mpg"].dropna()
data_auto["horsepower"] = data_auto["horsepower"].dropna()
plt.plot(data_auto["horsepower"], data_auto["mpg"], "ro")
plt.xlabel("Caballos de Potencia")
plt.ylabel("Consumo (millas por galeón)")
plt.title("CV vs MPG")


# ### Modelo de regresión lineal
# * mpg = a + b * horsepower

# In[39]:


X = data_auto["horsepower"].fillna(data_auto["horsepower"].mean()).to_numpy()
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())
X_data = X[:,np.newaxis]


# In[40]:


lm = LinearRegression()
lm.fit(X_data,Y)


# In[41]:


type(X)


# In[42]:


type(X_data)


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.plot(X, lm.predict(X_data), color="blue")


# In[44]:


lm.score(X_data, Y)


# In[45]:


SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean
SSD, RSE, y_mean, error*100


# ### Modelo de regresión cuadrático
# * mpg = a + b * horsepower^2 

# In[46]:


X_data = X**2
X_data = np.asarray(X_data)
X_data = X_data[:,np.newaxis]


# In[47]:


lm = LinearRegression()
lm.fit(X_data, Y)


# In[48]:


lm.score(X_data, Y)


# In[49]:


SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean
SSD, RSE, y_mean, error*100


# ### Modelo de regresión  lineal y cuadrático
# * mpg = a + b * horsepower + c * horsepower^2

# In[50]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


# In[51]:


poly = PolynomialFeatures(degree=2)


# In[52]:


X = np.asarray(X)
X_data = poly.fit_transform(X[:,np.newaxis])


# In[53]:


lm = linear_model.LinearRegression()
lm.fit(X_data, Y)


# In[54]:


lm.score(X_data, Y)


# In[55]:


lm.intercept_


# In[56]:


lm.coef_


# mpg = 55.026 -0.434 * hp + 0.00112615 * hp^2

# In[57]:


def regresion_validation(X_data, Y, Y_pred):
    SSD = np.sum((Y - Y_pred)**2)
    RSE = np.sqrt(SSD/(len(X_data)-1))
    y_mean = np.mean(Y)
    error = RSE/y_mean
    print("SSD: "+str(SSD)+", RSE: " +str(RSE) + ", Y_mean: " +str(y_mean) +", error: " + str(error*100)+ "%")


# In[58]:


for d in range(2,12):
    poly = PolynomialFeatures(degree=d)
    X_data = poly.fit_transform(X[:,np.newaxis])
    lm = linear_model.LinearRegression()
    lm.fit(X_data, Y)
    print("Regresión de grado "+str(d))
    print("R2:" +str(lm.score(X_data, Y)))
    print(lm.intercept_)
    print(lm.coef_)
    regresion_validation(X_data, Y, lm.predict(X_data))


# # El problema de los outliers

# In[59]:


plt.plot(data_auto["displacement"], data_auto["mpg"], "ro")


# In[60]:


X = data_auto["displacement"].fillna(data_auto["displacement"].mean()).to_numpy()
X = X[:,np.newaxis]
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())

lm = LinearRegression()
lm.fit(X, Y)


# In[61]:


lm.score(X,Y)


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.plot(X, lm.predict(X), color="blue")


# In[63]:


data_auto[(data_auto["displacement"]>250)&(data_auto["mpg"]>35)]


# In[64]:


data_auto[(data_auto["displacement"]>300)&(data_auto["mpg"]>20)]


# In[65]:


data_auto_clean = data_auto.drop([395, 258, 305, 372])


# In[69]:


X = data_auto_clean["displacement"].fillna(data_auto_clean["displacement"].mean()).to_numpy()
X = X[:,np.newaxis]
Y = data_auto_clean["mpg"].fillna(data_auto_clean["mpg"].mean())

lm = LinearRegression()
lm.fit(X, Y)


# In[70]:


lm.score(X,Y)


# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(X,Y, "ro")
plt.plot(X, lm.predict(X), color="blue")

