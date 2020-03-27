#!/usr/bin/env python
# coding: utf-8

# In[78]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


# In[86]:


files = glob.glob("imagenes/*.png")
n_files = len(files)

im = []
    
for ii in files: 
    i = plt.imread(ii)
    d= np.float_(i.flatten())
    im.append(d)

images = np.array(im)
#plt.imshow(images[1,:,:].reshape((100,100,3)))

n_clusters = 20


# In[89]:


#cluster = k_means.predict(images)
X_inertia = []
n_arboles = []


for i in range(n_clusters):
    
    k_means = sklearn.cluster.KMeans(n_clusters=(i+1))
    k_means.fit(images)
    X_inertia.append(k_means.inertia_)
    n_arboles.append(i+1)
    
inercia = np.array(X_inertia)
n_arboles = np.array(n_arboles)



# In[95]:


plt.figure()

plt.plot(n_arboles,X_inertia)
plt.xlabel("numero arboles")
plt.ylabel("inercia")
plt.savefig("inercia.png")


# In[161]:


n_clusters = 4
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(images)

cluster = k_means.predict(images)

# asigno a cada pixel el lugar del centro de su cluster
#X_centered = images.copy()

X_centered = []
for i in range(n_clusters):
    ii = cluster==i
    X_centered.append(k_means.cluster_centers_[i])
    
centros = np.array(X_centered)

centros_vector = centros.reshape((4,100,100,3))

images_reshape = images.reshape((87,100,100,3))

norma_1 = []
norma_2 = []
norma_3 = []
norma_4 = []

for i in range(87):
    producto = centros_vector[1,:,:]*images_reshape[i,:,:]    
    norma_1.append(np.linalg.norm(producto))
    
index_norma_1 = np.argsort(norma_1)





plt.figure(figsize=(10,5))
for i in range(1,6):
    plt.subplot(4,5,i)
    plt.imshow(images_reshape[(index_norma_1[i]),:,:])
    plt.subplot(4,5,i+5)
    plt.imshow(images_reshape[(index_norma_1[i+5]),:,:])
    plt.subplot(4,5,i+10)
    plt.imshow(images_reshape[(index_norma_1[i+10]),:,:])
    plt.subplot(4,5,i+15)
    plt.imshow(images_reshape[(index_norma_1[i+15]),:,:])
    
plt.savefig("ejemplo_clases.png")    
#data_centered = X_centered.reshape((87,100,100,3))



# devuelvo los datos a las dimensiones originales de la imagen
#data_centered = X_centered.reshape((100,100,3))

