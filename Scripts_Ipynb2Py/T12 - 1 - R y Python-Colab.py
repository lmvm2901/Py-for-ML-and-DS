#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T12%20-%201%20-%20R%20y%20Python-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Clonamos el repositorio para obtener los dataSet

# In[1]:


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


# # Juntando R y Python

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


get_ipython().system('pip install rpy2')
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri


# In[72]:


rpy2.robjects.numpy2ri.activate()


# In[73]:


codigo_r = """
saludar <- function(cadena){
    return(paste("Hola, ", cadena))
}
"""


# In[74]:


ro.r(codigo_r)


# In[75]:


saludar_py = ro.globalenv["saludar"]


# In[76]:


res = saludar_py("Antonio Banderas")
res[0]


# In[77]:


type(res)


# In[78]:


print(saludar_py.r_repr())


# In[79]:


var_from_python = ro.FloatVector(np.arange(1,5,0.1))


# In[80]:


var_from_python


# In[81]:


print(var_from_python.r_repr())


# In[15]:


ro.globalenv["var_to_r"] = var_from_python


# In[16]:


ro.r("var_to_r")


# In[17]:


ro.r("sum(var_to_r)")


# In[18]:


ro.r("mean(var_to_r)")


# In[19]:


ro.r("sd(var_to_r)")


# In[20]:


np.sum(var_from_python)


# In[21]:


np.mean(var_from_python)


# In[22]:


ro.r("summary(var_to_r)")


# In[23]:


ro.r("hist(var_to_r, breaks = 4)")


# # Trabajar de forma conjunta entre R y Python

# In[1]:


from rpy2.robjects.packages import importr


# In[ ]:


ro.r("install.packages('extRemes')")# si os falla decidle 'n' al hacer la instalación


# In[17]:


extremes = importr("extRemes") # library(extRemes)


# In[18]:


fevd = extremes.fevd


# In[19]:


print(fevd.__doc__)


# In[25]:


data = pd.read_csv("/content/python-ml-course/datasets/time/time_series.txt", 
                   sep = "\s*", skiprows = 1, parse_dates = [[0,1]],
                   names = ["date", "time", "wind_speed"],
                   index_col = 0)


# In[26]:


data.head(5)


# In[23]:


data.shape


# In[36]:


max_ws = data.wind_speed.groupby(pd.Grouper(freq="A")).max()


# In[37]:


max_ws


# In[38]:


max_ws.plot(kind="bar", figsize=(16,9))


# In[39]:


result = fevd(max_ws.values, type="GEV", method = "GMLE")


# In[40]:


print(type(result))


# In[41]:


result.r_repr


# In[43]:


print(result.names)


# In[44]:


res = result.rx("results")


# In[58]:


print(res[0])


# In[62]:


loc, scale, shape = res[0].rx("par")[0]


# In[63]:


loc


# In[64]:


scale


# In[65]:


shape


# # Función mágica para R

# In[66]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[67]:


help(rpy2.ipython.rmagic.RMagics.R)


# In[68]:


get_ipython().run_line_magic('R', 'X=c(1,4,5,7); sd(X); mean(X)')


# In[69]:


get_ipython().run_cell_magic('R', '', 'Y = c(2,4,3,9)\nlm = lm(Y~X)\nsummary(lm)')


# In[70]:


get_ipython().run_line_magic('R', '-i result plot.fevd(result)')


# In[82]:


get_ipython().run_line_magic('R', '-i var_from_python hist(var_from_python)')


# In[84]:


ro.globalenv["result"] = result
ro.r("plot.fevd(result)") ## puede dar error y generar un objeto rpy2.rinterface.NULL


# # Un ejemplo complejo de R, Python y Rmagic

# In[90]:


metodos = ["MLE", "GMLE", "Bayesian", "Lmoments"]
tipos = ["GEV", "Gumbel"]


# In[91]:


for t in tipos:
    for m in metodos:
        print("Tipo de Ajuste: ", t)
        print("Método del Ajuste: ", m)
        result = fevd(max_ws.values, method = m, type = t)
        print(result.rx("results")[0])
        get_ipython().run_line_magic('R', '-i result plot.fevd(result)')


# In[ ]:




