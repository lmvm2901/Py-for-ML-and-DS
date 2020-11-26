#!/usr/bin/env python
# coding: utf-8

# # Propagación de la afinidad

# In[1]:


from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs


# In[12]:


centers = [[1,1], [-1,-1], [1,-1]]
X, labels = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=None)


# In[13]:


import matplotlib.pyplot as plt
from itertools import cycle


# In[14]:


plt.scatter(X[:,0], X[:,1], c=labels, s = 5, cmap = "autumn")


# In[16]:


af = AffinityPropagation(preference=-50).fit(X)


# In[17]:


cluster_center_ids = af.cluster_centers_indices_


# In[18]:


labels = af.labels_


# In[19]:


n_clust = len(cluster_center_ids)
n_clust


# In[20]:


def report_affinity_propagation(X):
    af = AffinityPropagation(preference=-50).fit(X)
    cluster_center_ids = af.cluster_centers_indices_
    n_clust = len(cluster_center_ids)
    clust_labels = af.labels_
    
    print("Número estimado de clusters: %d"%n_clust)
    print("Homogeneidad: %0.3f"%metrics.homogeneity_score(labels, clust_labels))
    print("Completitud: %0.3f"%metrics.completeness_score(labels, clust_labels))
    print("V-measure: %0.3f"%metrics.v_measure_score(labels, clust_labels))
    print("R2 ajustado: %0.3f"%metrics.adjusted_rand_score(labels, clust_labels))
    print("Información mútua ajustada: %0.3f"%metrics.adjusted_mutual_info_score(labels, clust_labels))
    print("Coeficiente de la silueta: %0.3f"%metrics.silhouette_score(X, labels, metric="sqeuclidean"))
    
    plt.figure(figsize=(16,9))
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clust), colors):
        class_members = (clust_labels==k)
        clust_center = X[cluster_center_ids[k]]
        plt.plot(X[class_members,0], X[class_members, 1], col +'.')
        plt.plot(clust_center[0], clust_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([clust_center[0],x[0]], [clust_center[1], x[1]], col)
            
    plt.title("Número estimado de clusters %d"%n_clust)
    plt.show()


# In[21]:


report_affinity_propagation(X)


# In[22]:


plt.hist(af.labels_)

