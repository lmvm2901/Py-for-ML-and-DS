#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T10%20-%201%20-%20Analisis%20de%20Componentes%20Principales-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Análisis de Componentes Principales - Paso a Paso

# * Estandarizar los datos (para cada una de las m observaciones)
# * Obtener los vectores y valores propios a partir de la matriz de covarianzas o de correlaciones o incluso la técnica de singular vector decomposition.
# * Ordenar los valores propios en orden descendente y quedarnos con los *p* que se correpondan a los *p* mayores y así disminuir el número de variables del dataset (p<m)
# * Constrir la matriz de proyección W a partir de los p vectores propios
# * Transformar el dataset original X a través de W para así obtener dadtos en el subespacio dimensional de dimensión *p*, que será Y

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("/content/python-ml-course/datasets/iris/iris.csv")


# In[ ]:


df.head()


# In[ ]:


X = df.iloc[:,0:4].values
y = df.iloc[:,4].values


# In[ ]:


X[0]


# In[ ]:


import chart_studio.plotly as py
import plotly.graph_objects as go
import chart_studio


# In[ ]:


chart_studio.tools.set_credentials_file(username='JuanGabriel', api_key='6mEfSXf8XNyIzpxwb8z7')


# In[ ]:


traces = []
legend = {0:True, 1:True, 2:True, 3:True}

colors = {'setosa': 'rgb(255,127,20)',
         'versicolor': 'rgb(31, 220, 120)',
         'virginica': 'rgb(44, 50, 180)'}


for col in range(4): 
    for key in colors:
        traces.append(go.Histogram(x=X[y==key, col],
                                   opacity = 0.7, 
                                   xaxis="x%s"%(col+1),
                                   marker={"color":colors[key]},
                                   name = key, showlegend=legend[col])
                     )
        
    legend = {0:False, 1:False, 2:False, 3:False}

layout = go.Layout(
    title={"text":"Distribución de los rasgos de las diferentes flores Iris",
           "xref" : "paper","x" : 0.5},
    barmode="overlay",
    xaxis= {"domain" : [0,0.25], "title":"Long. Sépalos (cm)"},
    xaxis2= {"domain" : [0.3, 0.5], "title" : "Anch. Sépalos (cm)"},
    xaxis3= {"domain" : [0.55, 0.75], "title" : "Long. Pétalos (cm)"},
    xaxis4= {"domain" : [0.8,1.0], "title" : "Anch. Pétalos (cm)"},
    yaxis={"title":"Número de ejemplares"}
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
#fig.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


X_std = StandardScaler().fit_transform(X)


# In[ ]:


traces = []
legend = {0:True, 1:True, 2:True, 3:True}

colors = {'setosa': 'rgb(255,127,20)',
         'versicolor': 'rgb(31, 220, 120)',
         'virginica': 'rgb(44, 50, 180)'}


for col in range(4): 
    for key in colors:
        traces.append(go.Histogram(x=X_std[y==key, col],
                                   opacity = 0.7, 
                                   xaxis="x%s"%(col+1),
                                   marker={"color":colors[key]},
                                   name = key, showlegend=legend[col])
                     )
        
    legend = {0:False, 1:False, 2:False, 3:False}

layout = go.Layout(
    title={"text":"Distribución de los rasgos de las diferentes flores Iris",
           "xref" : "paper","x" : 0.5},
    barmode="overlay",
    xaxis= {"domain" : [0,0.25], "title":"Long. Sépalos (cm)"},
    xaxis2= {"domain" : [0.3, 0.5], "title" : "Anch. Sépalos (cm)"},
    xaxis3= {"domain" : [0.55, 0.75], "title" : "Long. Pétalos (cm)"},
    xaxis4= {"domain" : [0.8,1.0], "title" : "Anch. Pétalos (cm)"},
    yaxis={"title":"Número de ejemplares"}
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
#fig.show()


# ### 1- Calculamos la descomposición de valores y vectores propios
# ##### a) Usando la Matriz de Covarianzas

# In[ ]:


from IPython.display import display, Math, Latex


# In[ ]:


display(Math(r'\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^m (x_{ij} - \overline{x_j})(x_{ik} - \overline{x_k})'))


# In[ ]:


display(Math(r'\Sigma = \frac{1}{n-1}((X-\overline{x})^T(X-\overline{x}))'))


# In[ ]:


display(Math(r'\overline{x} = \sum_{i=1}^n x_i\in \mathbb R^m'))


# In[ ]:


import numpy as np


# In[ ]:


mean_vect = np.mean(X_std, axis=0)
mean_vect


# In[ ]:


cov_matrix = (X_std - mean_vect).T.dot((X_std - mean_vect))/(X_std.shape[0]-1)
print("La matriz de covarianzas es \n%s"%cov_matrix)


# In[ ]:


np.cov(X_std.T)


# In[ ]:


eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print("Valores propios \n%s"%eig_vals)
print("Vectores propios \n%s"%eig_vectors)


# ##### b) Usando la Matriz de Correlaciones

# In[ ]:


corr_matrix = np.corrcoef(X_std.T)
corr_matrix


# In[ ]:


eig_vals_corr, eig_vectors_corr = np.linalg.eig(corr_matrix)
print("Valores propios \n%s"%eig_vals_corr)
print("Vectores propios \n%s"%eig_vectors_corr)


# In[ ]:


corr_matrix = np.corrcoef(X.T)
corr_matrix


# ##### c) Singular Value Decomposition

# In[ ]:


u,s,v = np.linalg.svd(X_std.T)
u


# In[ ]:


s


# In[ ]:


v


# ### 2 - Las componentes principales

# In[ ]:


for ev in eig_vectors:
    print("La longitud del VP es: %s"%np.linalg.norm(ev))


# In[ ]:


eigen_pairs = [(np.abs(eig_vals[i]), eig_vectors[:,i]) for i in range(len(eig_vals))]
eigen_pairs


# Ordenamos los vectores propios con valor propio de mayor a menor

# In[ ]:


eigen_pairs.sort()
eigen_pairs.reverse()
eigen_pairs


# In[ ]:


print("Valores propios en orden descendente:")
for ep in eigen_pairs:
    print(ep[0])


# In[ ]:


total_sum = sum(eig_vals)
var_exp = [(i/total_sum)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[ ]:


plot1 = go.Bar(x=[f"CP {i}" for i in range(1,5)], y=var_exp, showlegend= True)
plot2 = go.Scatter(x=[f"CP {i}" for i in range(1,5)], y=cum_var_exp, showlegend= True)

data = [plot1,plot2]

layout = go.Layout(xaxis= {"title": "Componentes principales"},
                  yaxis ={"title": "Porcentaje de varianza explicada"},
                  title = "Porcentaje de variabilidad explicada por cada componente principal")

fig = go.Figure(data=data,layout=layout)
py.iplot(fig)
#fig.show()


# In[ ]:


W = np.hstack((eigen_pairs[0][1].reshape(4,1), 
               eigen_pairs[1][1].reshape(4,1)))
W


# In[ ]:


X[0]


# ### 3- Proyectando las variables en el nuevo subespacio vectorial

# In[ ]:


display(Math(r'Y = X \cdot W, X \in M(\mathbb R)_{150, 4}, W \in M(\mathbb R)_{4,2}, Y \in M(\mathbb R)_{150, 2}'))


# In[ ]:


Y = X_std.dot(W)
Y


# In[ ]:


results = []
for name in ('setosa', 'versicolor', 'virginica'):
    result = go.Scatter(x= Y[y==name,0], y =Y[y==name, 1],
                       mode = "markers", name=name,
    marker= { "size": 12, "line" : { "color" : 'rgba(220,220,220,0.15)', "width":0.5},
           "opacity": 0.8})
    results.append(result)
    
layout = go.Layout(showlegend = True, 
                   scene ={ "xaxis" :{"title": "Componente Principal 1"},
                            "yaxis" : {"title": "Componente Principal 2"}},
                  xaxis ={ "zerolinecolor": "gray"},
                  yaxis={ "zerolinecolor": "gray"})
fig = go.Figure(data=results,layout=layout)
py.iplot(fig)
#fig.show()

