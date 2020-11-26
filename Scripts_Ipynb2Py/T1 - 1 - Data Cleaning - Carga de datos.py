import pandas as pd
import os

# In[ ]:


mainpath = "/content/drive/My Drive/Curso Machine Learning con Python/datasets/"
filename = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filename)

# In[ ]:


data = pd.read_csv(fullpath)

# In[ ]:


data.head()

# ### Ejemplos de los parámetros de la función read_csv
# ```
# read.csv(filepath="/Users/JuanGabriel/Developer/AnacondaProjects/python-ml-course/datasets/titanic/titanic3.csv",
#         sep = ",", 
#         dtype={"ingresos":np.float64, "edad":np.int32}, 
#         header=0,names={"ingresos", "edad"},
#         skiprows=12, index_col=None, 
#         skip_blank_lines=False, na_filter=False
#         )
# ```

# In[ ]:


data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", sep=",")  # CUIDADO: ES EL TXT;
# NO EL CSV


# In[ ]:


data2.head()

# In[ ]:


data2.columns.values

# In[ ]:


data_cols = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Columns.csv")
data_col_list = data_cols["Column_Names"].tolist()
data2 = pd.read_csv(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt",
                    header=None, names=data_col_list)
data2.columns.values

# # Carga de datos a través de la función open

# In[ ]:


data3 = open(mainpath + "/" + "customer-churn-model/Customer Churn Model.txt", 'r')

# In[ ]:


cols = data3.readline().strip().split(",")
n_cols = len(cols)

# In[ ]:


counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []

# In[ ]:


for line in data3:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1

print("El data set tiene %d filas y %d columnas" % (counter, n_cols))

# In[ ]:


df3 = pd.DataFrame(main_dict)
df3.head()

# ## Lectura y escritura de ficheros

# In[ ]:


infile = mainpath + "/" + "customer-churn-model/Customer Churn Model.txt"
outfile = mainpath + "/" + "customer-churn-model/Table Customer Churn Model.txt"

# In[ ]:


with open(infile, "r") as infile1:
    with open(outfile, "w") as outfile1:
        for line in infile1:
            fields = line.strip().split(",")
            outfile1.write("\t".join(fields))
            outfile1.write("\n")

# In[ ]:


df4 = pd.read_csv(outfile, sep="\t")
df4.head()

# # Leer datos desde una URL

# In[ ]:


medals_url = "http://winterolympicsmedals.com/medals.csv"

# In[ ]:


medals_data = pd.read_csv(medals_url)

# In[ ]:


medals_data.head()


# #### Ejercicio de descarga de datos con urllib3 Vamos a hacer un ejemplo usando la librería urllib3 para leer los
# datos desde una URL externa, procesarlos y convertirlos a un data frame de *python* antes de guardarlos en un CSV
# local.

# In[ ]:


def downloadFromURL(url, filename, sep=",", delim="\n", encoding="utf-8",
                    mainpath="/content/drive/My Drive/Curso Machine Learning con Python/datasets"):
    # primero importamos la librería y hacemos la conexión con la web de los datos
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    print("El estado de la respuesta es %d" % r.status)
    response = r.data
    # CORREGIDO: eliminado un doble decode que daba error

    # El objeto reponse contiene un string binario, así que lo convertimos a un string descodificándolo en UTF-8
    str_data = response.decode(encoding)

    # Dividimos el string en un array de filas, separándolo por intros
    lines = str_data.split(delim)

    # La primera línea contiene la cabecera, así que la extraemos
    col_names = lines[0].split(sep)
    n_cols = len(col_names)

    # Generamos un diccionario vacío donde irá la información procesada desde la URL externa
    counter = 0
    main_dict = {}
    for col in col_names:
        main_dict[col] = []

    # Procesamos fila a fila la información para ir rellenando el diccionario con los datos como hicimos antes
    for line in lines:
        # Nos saltamos la primera línea que es la que contiene la cabecera y ya tenemos procesada
        if (counter > 0):
            # Dividimos cada string por las comas como elemento separador
            values = line.strip().split(sep)
            # Añadimos cada valor a su respectiva columna del diccionario
            for i in range(len(col_names)):
                main_dict[col_names[i]].append(values[i])
        counter += 1

    print("El data set tiene %d filas y %d columnas" % (counter, n_cols))

    # Convertimos el diccionario procesado a Data Frame y comprobamos que los datos son correctos
    df = pd.DataFrame(main_dict)
    print(df.head())

    # Elegimos donde guardarlo (en la carpeta athletes es donde tiene más sentido por el contexto del análisis)
    fullpath = os.path.join(mainpath, filename)

    # Lo guardamos en CSV, en JSON o en Excel según queramos
    df.to_csv(fullpath + ".csv")
    df.to_json(fullpath + ".json")
    df.to_excel(fullpath + ".xls")
    print("Los ficheros se han guardado correctamente en: " + fullpath)

    return df


# In[ ]:


medals_df = downloadFromURL(medals_url, "athletes/downloaded_medals")
medals_df.head()

# ## Ficheros XLS y XLSX

# In[ ]:


mainpath = "/content/drive/My Drive/Curso Machine Learning con Python/datasets"
filename = "titanic/titanic3.xls"

# In[ ]:


titanic2 = pd.read_excel(mainpath + "/" + filename, "titanic3")

# In[ ]:


titanic3 = pd.read_excel(mainpath + "/" + filename, "titanic3")

# In[ ]:


titanic3.to_csv(mainpath + "/titanic/titanic_custom.csv")

# In[ ]:


titanic3.to_excel(mainpath + "/titanic/titanic_custom.xls")

# In[ ]:


titanic3.to_json(mainpath + "/titanic/titanic_custom.json")
