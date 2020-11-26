#!/usr/bin/env python
# coding: utf-8

# # Árbol de decisión para especies de flores

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("../datasets/iris/iris.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


plt.hist(data.Species)


# In[ ]:


data.Species.unique()


# In[ ]:


colnames = data.columns.values.tolist()
predictors = colnames[:4]
target = colnames[4]


# In[ ]:


import numpy as np


# In[ ]:


data["is_train"] = np.random.uniform(0,1, len(data))<=0.75


# In[ ]:


data["is_train"] = (data["is_train"].astype(np.float32) - data["is_train"].astype(np.float32)).astype(np.bool)
plt.hist(data.is_train)


# In[ ]:


train, test = data[data["is_train"]==True], data[data["is_train"]==False]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=20, random_state=99)
tree.fit(train[predictors], train[target])


# In[ ]:


preds = tree.predict(test[predictors])


# In[ ]:


pd.crosstab(test[target], preds, rownames=["Actual"], colnames=["Predictions"])


# ## Visualización del árbol de decisión

# In[ ]:


from sklearn.tree import export_graphviz


# In[ ]:


with open("resources/iris_dtree.dot", "w") as dotfile:
    export_graphviz(tree, out_file=dotfile, feature_names=predictors)
    dotfile.close()


# In[ ]:


import os
from graphviz import Source


# In[ ]:


file = open("resources/iris_dtree.dot", "r")
text = file.read()
text


# In[ ]:


Source(text)


# ## Cross Validation para la poda

# In[ ]:


X = data[predictors]
Y = data[target]


# In[ ]:


tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=20, random_state=99)
tree.fit(X,Y)


# In[ ]:


from sklearn.cross_validation import KFold


# In[ ]:


cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=1)


# In[ ]:


from sklearn.cross_validation import cross_val_score


# In[ ]:


scores = cross_val_score(tree, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
scores


# In[ ]:


score = np.mean(scores)
score


# In[ ]:


for i in range(1,11):
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_split=20, random_state=99)
    tree.fit(X,Y)
    cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=1)
    scores = cross_val_score(tree, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
    score = np.mean(scores)
    print("Score para i = ",i," es de ", score)
    print("   ",tree.feature_importances_)


# In[ ]:


predictors


# ## Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forest = RandomForestClassifier(n_jobs=2, oob_score=True, n_estimators=100)
forest.fit(X,Y)


# In[ ]:


forest.oob_decision_function_


# In[ ]:


forest.oob_score_


# In[ ]:




