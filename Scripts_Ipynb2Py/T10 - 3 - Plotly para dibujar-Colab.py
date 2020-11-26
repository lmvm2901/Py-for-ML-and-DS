#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/joanby/python-ml-course/blob/master/notebooks/T10%20-%203%20-%20Plotly%20para%20dibujar-Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


# # Gráficos con PlotLy

# In[1]:


get_ipython().system('pip install chart_studio')
import chart_studio.plotly as py
import plotly.graph_objects as go
from chart_studio import tools as tls

tls.set_credentials_file(username='JuanGabriel', api_key='6mEfSXf8XNyIzpxwb8z7')


# In[2]:


import plotly
plotly.__version__


# In[5]:


help(plotly)


# In[8]:


import numpy as np
help(np.random)


# # Scatter Plots sencillos

# In[10]:


N = 2000
random_x = np.random.randn(N)
random_y = np.random.randn(N)


# In[11]:


trace = go.Scatter(x = random_x, y = random_y, mode = "markers")


# In[12]:


py.iplot([trace], filename = "basic-scatter")


# In[13]:


plot_url = py.plot([trace], filename = "basic-scatter-inline")
plot_url


# # Gráficos combinados
# 

# In[14]:


N = 200
rand_x = np.linspace(0,1, N)
rand_y0 = np.random.randn(N) + 3
rand_y1 = np.random.randn(N)
rand_y2 = np.random.randn(N) - 3


# In[15]:


trace0 = go.Scatter(x = rand_x, y = rand_y0, mode="markers", name="Puntos")
trace1 = go.Scatter(x = rand_x, y = rand_y1, mode="lines", name="Líneas")
trace2 = go.Scatter(x = rand_x, y = rand_y2, mode="lines+markers", name="Puntos y líneas")
data = [trace0, trace1, trace2]


# In[16]:


py.iplot(data, filename = "scatter-line-plot")


# # Estilizado de gráficos

# In[17]:


trace = go.Scatter(x = random_x, y = random_y, name = "Puntos de estilo guay", mode="markers",
                  marker = dict(size = 12, color = "rgba(140,20,20,0.8)", line = dict(width=2, color="rgb(10,10,10)")))


# In[18]:


layout = dict(title = "Scatter Plot Estilizado", xaxis = dict(zeroline = False), yaxis = dict(zeroline=False))


# In[19]:


fig = dict(data = [trace], layout = layout)
py.iplot(fig)


# In[22]:


trace = go.Scatter(x = random_x, y = random_y, name = "Puntos de estilo guay", mode="markers",
                  marker = dict(size = 8, color = "rgba(10,80,220,0.25)", line = dict(width=1, color="rgb(10,10,80)")))


fig = dict(data = [trace], layout = layout)
py.iplot(fig)


# In[23]:


trace = go.Histogram(x = random_x, name = "Puntos de estilo guay")


fig = dict(data = [trace], layout = layout)
py.iplot(fig)


# In[30]:


trace = go.Box(x = random_x, name = "Puntos de estilo guay", fillcolor = "rgba(180,25,95,0.6)")


fig = dict(data = [trace], layout = layout)
py.iplot(fig,  filename = "basic-scatter-inline")


# In[28]:


help(go.Box)


# # Información al hacer Hover

# In[32]:


import pandas as pd
data = pd.read_csv("/content/python-ml-course/datasets/usa-population/usa_states_population.csv")


# In[33]:


data


# In[5]:


N = 53
c = ['hsl('+str(h)+', 50%, 50%)' for h in np.linspace(0,360,N)]


# In[8]:


l = []
y = []
for i in range(int(N)):
    y.append((2000+i))
    trace0 = go.Scatter(
        x = data["Rank"], 
        y = data["Population"]+ i*1000000,
        mode = "markers",
        marker = dict(size = 14, line = dict(width=1), color = c[i], opacity = 0.3),
        name = data["State"]
    )
    l.append(trace0)
    
    


# In[ ]:





# In[38]:


layout = go.Layout(title = "Población de los estados de USA",
                  hovermode = "closest", 
                  xaxis = dict(title="ID", ticklen=5, zeroline=False, gridwidth=2),
                  yaxis = dict(title="Población", ticklen=5, gridwidth=2),
                  showlegend = False)


# In[39]:


fig = go.Figure(data = l, layout = layout)
py.iplot(fig, filename = "basic-scatter-inline")


# In[40]:


trace = go.Scatter(y = np.random.randn(1000),
                  mode = "markers", marker = dict(size = 16, color = np.random.randn(1000), 
                                                  colorscale = "Viridis", showscale=True))


# In[42]:


py.iplot([trace],  filename = "basic-scatter-inline")


# # Datasets muy grandes

# In[43]:


N = 100000
trace = go.Scattergl(x = np.random.randn(N), y = np.random.randn(N), mode = "markers",
                    marker = dict(color="#BAD5FF", line = dict(width=1)))


# In[44]:


py.iplot([trace],  filename = "basic-scatter-inline")


# In[ ]:




