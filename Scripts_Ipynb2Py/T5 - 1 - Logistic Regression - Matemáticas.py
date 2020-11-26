#!/usr/bin/env python
# coding: utf-8

# # Las matemáticas tras la regresión logística

# ### Las tablas de contingencia

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("../datasets/gender-purchase/Gender Purchase.csv")
df.head()


# In[3]:


df.shape


# In[4]:


contingency_table = pd.crosstab(df["Gender"], df["Purchase"])
contingency_table


# In[5]:


contingency_table.sum(axis = 1)


# In[6]:


contingency_table.sum(axis = 0)


# In[7]:


contingency_table.astype("float").div(contingency_table.sum(axis=1), axis = 0)


# ### La probabilidad condicional

# In[8]:


from IPython.display import display, Math, Latex


# * ¿Cuál es la probabilidad de que un cliente compre un producto sabiendo que es un hombre?
# * ¿Cuál es la probabilidad de que sabiendo que un cliente compra un producto sea mujer?

# In[9]:


display(Math(r'P(Purchase|Male) = \frac{Numero\ total\ de\ compras\ hechas\ por\ hombres}{Numero\ total\ de\ hombres\ del\ grupo} = \frac{Purchase\cap Male}{Male}'))
121/246


# In[10]:


display(Math(r'P(No\ Purchase|Male) = 1-P(Purchase|Male)'))
125/246


# In[11]:


display(Math(r'P(Female|Purchase) = \frac{Numero\ total\ de\ compras\ hechas\ por\ mujeres}{Numero\ total\ de\ compras} = \frac{Female\cap Purchase}{Purchase}'))
159/280


# In[12]:


display(Math(r'P(Male|Purchase)'))
121/280


# In[13]:


display(Math(r'P(Purchase|Male)'))
print(121/246)
display(Math(r'P(NO\ Purchase|Male)'))
print(125/246)
display(Math(r'P(Purchase|Female)'))
print(159/265)
display(Math(r'P(NO\ Purchase|Female)'))
print(106/265)


# ### Ratio de probabilidades
# Cociente entre los casos de éxito sobre los de fracaso en el suceso estudiado y para cada grupo

# In[14]:


display(Math(r'P_m = \ probabilidad\ de\ hacer\ compra\ sabiendo\ que\ es \ un \ hombre'))

display(Math(r'P_f = \ probabilidad\ de\ hacer\ compra\ sabiendo\ que\ es \ una\ mujer'))

display(Math(r'odds\in[0,+\infty]'))

display(Math(r'odds_{purchase,male} = \frac{P_m}{1-P_m} = \frac{N_{p,m}}{N_{\bar p, m}}'))

display(Math(r'odds_{purchase,female} = \frac{P_F}{1-P_F} = \frac{N_{p,f}}{N_{\bar p, f}}'))


# In[15]:


pm = 121/246
pf = 159/265
odds_m = pm/(1-pm)# 121/125
odds_f = pf/(1-pf)# 159/106


# In[16]:


odds_m


# In[17]:


odds_f


# * Si el ratio es superior a 1, es más probable el éxito que el fracas. Cuanto mayor es el ratio, más probabilidad de éxito en nuestro suceso.
# * Si el ratio es exactamente igual a 1, éxito y fracaso son equiprobables (p=0.5)
# * Si el ratio es menor que 1, el fracaso es más probable que el éxito. Cuanto menor es el ratio, menor es la probabilidad de éxito del suceso.

# In[18]:


display(Math(r'odds_{ratio} = \frac{odds_{purchase,male}}{odds_{purchase,female}}'))


# In[19]:


odds_r = odds_m/odds_f


# In[20]:


odds_r


# In[21]:


1/odds_r# odds_f/odds_m


# ### La regresión logística desde la regresión lineal

# In[22]:


display(Math(r'y = \alpha + \beta \cdot x'))
display(Math(r'(x,y)\in[-\infty, +\infty]^2'))


# In[23]:


display(Math(r'Y\in\{0,1\}??'))
display(Math(r'P\in [0,1]'))
display(Math(r'X\in [-\infty,\infty]'))

display(Math(r'P = \alpha + \beta\cdot X'))


# P es la probabilidad condicionada de éxito o de fracaso condicionada a la presencia de la variable X

# In[24]:


display(Math(r'\frac{P}{1-P} = \alpha + \beta\cdot X\in [0,+\infty]'))


# In[25]:


display(Math(r' ln(\frac{P}{1-P}) = \alpha + \beta\cdot X'))


# In[26]:


display(Math(r'\begin{cases}\frac{P}{1-P}\in[0,1]\Rightarrow ln(\frac{P}{1-P})\in[-\infty,0]\\ \frac{P}{1-P}\in[1,+\infty]\Rightarrow ln(\frac{P}{1-P})\in[0, \infty]\end{cases}'))


# In[27]:


display(Math(r' ln(\frac{P}{1-P}) = \alpha + \beta\cdot X'))
display(Math(r' \frac{P}{1-P} = e^{\alpha + \beta\cdot X}'))
display(Math(r' P = \frac{e^{\alpha+\beta\cdot X}}{1+e^{\alpha+\beta\cdot X}}'))
display(Math(r' P = \frac{1}{1+e^{-(\alpha+\beta\cdot X)}}'))


# * Si a+bX es muy pequeño (negativo), entonces P tiende a 0
# * Si a+bX = 0, P = 0.5
# * Si a+bX es muy grande (positivo), entonces P tiende a 1

# ### Regresión logística múltiple

# In[28]:


display(Math(r' P = \frac{1}{1+e^{-(\alpha+\sum_{i=1}^n\beta_i\cdot x_i)}}'))


# In[29]:


display(Math(r' \vec{\beta} = (\beta_1,\beta_2,\cdots,\beta_n)'))
display(Math(r' \vec{X} = (x_1,x_2,\cdots,x_n)'))
display(Math(r' P = \frac{1}{1+e^{-(\alpha+\vec{\beta_i}\cdot \vec{X})}}'))

