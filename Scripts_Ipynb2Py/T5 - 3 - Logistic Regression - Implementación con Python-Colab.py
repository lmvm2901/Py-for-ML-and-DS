#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T5%20-%203%20-%20Logistic%20Regression%20-%20Implementación%20con%20Python-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Regresión logística para predicciones bancarias

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("/content/python-ml-course/datasets/bank/bank.csv", sep=";")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns.values


# In[6]:


data["y"] = (data["y"]=="yes").astype(int)


# In[7]:


data.tail()


# In[8]:


data["education"].unique()


# In[9]:


data["education"] = np.where(data["education"]=="basic.4y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.6y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.9y", "Basic", data["education"])

data["education"] = np.where(data["education"]=="high.school", "High School", data["education"])
data["education"] = np.where(data["education"]=="professional.course", "Professional Course", data["education"])
data["education"] = np.where(data["education"]=="university.degree", "University Degree", data["education"])

data["education"] = np.where(data["education"]=="illiterate", "Illiterate", data["education"])
data["education"] = np.where(data["education"]=="unknown", "Unknown", data["education"])


# In[10]:


data["education"].unique()


# In[11]:


data["y"].value_counts()


# In[12]:


data.groupby("y").mean()


# In[13]:


data.groupby("education").mean()


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.education, data.y).plot(kind="bar")
plt.title("Frecuencia de compra en función del nivel de educación")
plt.xlabel("Nivel de educación")
plt.ylabel("Frecuencia de compra del producto")


# In[15]:


table=pd.crosstab(data.marital, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Diagrama apilado de estado civil contra el nivel de compras")
plt.xlabel("Estado civil")
plt.ylabel("Proporción de clientes")


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
table= pd.crosstab(data.day_of_week, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Frecuencia de compra en función del día de la semana")
plt.xlabel("Día de la semana")
plt.ylabel("Frecuencia de compra del producto")


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
table= pd.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Frecuencia de compra en función del mes")
plt.xlabel("Mes del año")
plt.ylabel("Frecuencia de compra del producto")


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
table.plot(kind="bar", stacked=False)
plt.title("Frecuencia de compra en función del mes")
plt.xlabel("Mes del año")
plt.ylabel("Frecuencia de compra del producto")


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.age.hist()
plt.title("Histograma de la Edad")
plt.xlabel("Edad")
plt.ylabel("Cliente")


# In[20]:


pd.crosstab(data.age, data.y).plot(kind="bar")


# In[21]:


pd.crosstab(data.poutcome, data.y).plot(kind="bar")


# ### Conversión de las variables categóricas a dummies

# In[22]:


categories = ["job", "marital", "education", "housing", "loan", "contact", 
              "month", "day_of_week", "poutcome"]
for category in categories:
    cat_list = "cat"+ "_"+category
    cat_dummies = pd.get_dummies(data[category], prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new


# In[23]:


data_vars = data.columns.values.tolist()


# In[24]:


to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in to_keep if v not in ["default"]]


# In[25]:


bank_data = data[to_keep]
bank_data.columns.values


# In[26]:


bank_data_vars = bank_data.columns.values.tolist()
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]


# ### Selección de rasgos para el modelo

# In[27]:


n = 12


# In[28]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[29]:


lr = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)


# In[30]:


rfe = RFE(lr, n_features_to_select=12)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())


# In[31]:


print(rfe.support_)


# In[32]:


print(rfe.ranking_)


# In[33]:


z=zip(bank_data_vars,rfe.support_, rfe.ranking_)


# In[34]:


list(z)


# In[35]:


cols = ["previous", "euribor3m", "job_blue-collar", "job_retired", "month_aug", "month_dec", 
        "month_jul", "month_jun", "month_mar", "month_nov", "day_of_week_wed", "poutcome_nonexistent"]


# In[36]:


X = bank_data[cols]
Y = bank_data["y"]


# ### Implementación del modelo en Python con statsmodel.api

# In[37]:


import statsmodels.api as sm


# In[38]:


logit_model = sm.Logit(Y, X)


# In[39]:


result = logit_model.fit()


# In[40]:


result.summary2()


# ### Implementación del modelo en Python con scikit-learn

# In[41]:


from sklearn import linear_model


# In[42]:


logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)


# In[43]:


logit_model.score(X,Y)


# In[44]:


1-Y.mean()


# In[45]:


pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_))))


# ## Validación del modelo logístico

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)


# In[48]:


lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)


# In[49]:


from IPython.display import display, Math, Latex


# In[50]:


display(Math(r'Y_p=\begin{cases}0& si\ p\leq0.5\\1&si\ p >0.5\end{cases}'))


# In[51]:


probs = lm.predict_proba(X_test)


# In[52]:


probs


# In[53]:


prediction = lm.predict(X_test)


# In[54]:


prediction


# In[55]:


display(Math(r'\varepsilon\in (0,1), Y_p=\begin{cases}0& si\ p\leq \varepsilon\\1&si\ p >\varepsilon\end{cases}'))


# In[56]:


prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df["prediction"] = np.where(prob_df[0]>threshold, 1, 0)
prob_df.head()


# In[57]:


pd.crosstab(prob_df.prediction, columns="count")


# In[58]:


390/len(prob_df)*100


# In[59]:


threshold = 0.15
prob_df["prediction"] = np.where(prob_df[0]>threshold, 1, 0)
pd.crosstab(prob_df.prediction, columns="count")


# In[60]:


331/len(prob_df)*100


# In[61]:


threshold = 0.05
prob_df["prediction"] = np.where(prob_df[0]>threshold, 1, 0)
pd.crosstab(prob_df.prediction, columns="count")


# In[62]:


732/len(prob_df)*100


# In[63]:


from sklearn import metrics


# In[64]:


metrics.accuracy_score(Y_test, prediction)


# ## Validación cruzada

# In[65]:


from sklearn.model_selection import cross_val_score


# In[66]:


scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring="accuracy", cv=10)


# In[67]:


scores


# In[68]:


scores.mean()


# ## Matrices de Confusión y curvas ROC

# In[69]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)


# In[70]:


lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)


# In[71]:


probs = lm.predict_proba(X_test)


# In[72]:


prob=probs[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df["prediction"] = np.where(prob_df[0]>=threshold, 1, 0)
prob_df["actual"] = list(Y_test)
prob_df.head()


# In[73]:


confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)


# In[74]:


TN=confusion_matrix[0][0]
TP=confusion_matrix[1][1]
FN=confusion_matrix[0][1]
FP=confusion_matrix[1][0]


# In[75]:


sens = TP/(TP+FN)
sens


# In[76]:


espc_1 = 1-TN/(TN+FP)
espc_1


# In[77]:


thresholds = [0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.3, 0.4, 0.5]
sensitivities = [1]
especifities_1 = [1]

for t in thresholds:
    prob_df["prediction"] = np.where(prob_df[0]>=t, 1, 0)
    prob_df["actual"] = list(Y_test)
    prob_df.head()

    confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)
    TN=confusion_matrix[0][0]
    TP=confusion_matrix[1][1]
    FP=confusion_matrix[0][1]
    FN=confusion_matrix[1][0]
    
    sens = TP/(TP+FN)
    sensitivities.append(sens)
    espc_1 = 1-TN/(TN+FP)
    especifities_1.append(espc_1)

sensitivities.append(0)
especifities_1.append(0)


# In[78]:


sensitivities


# In[79]:


especifities_1


# In[80]:


import matplotlib.pyplot as plt


# In[81]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(especifities_1, sensitivities, marker="o", linestyle="--", color="r")
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y)
plt.xlabel("1-Especifidad")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC")


# In[88]:


#HAY QUE ESPERAR QUE ACTUALICE GGPLOT LAS LIBRERIAS, SINO HAY QUE MODIFICAR ARCHIVOS INTERNOS


# In[1]:


get_ipython().system("pip install 'plotnine[all]'")
from sklearn import metrics
from pandas import Timestamp
from plotnine import *


# In[ ]:


espc_1, sensit, _ = metrics.roc_curve(Y_test, prob)


# In[83]:


df = pd.DataFrame({
    "esp":espc_1,
    "sens":sensit
})


# In[84]:


df.head()


# In[85]:


ggplot(df, aes(x="esp", y="sens")) +geom_line() + geom_abline(linetype="dashed")+xlim(-0.01,1.01)+ylim(-0.01,1.01)+xlab("1-Especifidad")+ylab("Sensibilidad")


# In[86]:


auc = metrics.auc(espc_1, sensit)
auc


# In[87]:


ggplot(df, aes(x="esp", y="sens")) + geom_area(alpha=0.25)+geom_line(aes(y="sens"))+ggtitle("Curva ROC y AUC=%s"%str(auc))

