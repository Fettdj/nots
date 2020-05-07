#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn.datasets import load_boston


# In[5]:


#Cargamos la libreria de boston
boston = load_boston()


# In[6]:


print(boston.DESCR)


# FORMULA MINIMIZAR EL ERROR CUADRATICO MEDIO: $\beta = (X^{T}X)^{-1}X^{T}Y$

# In[18]:


X = np.array(boston.data[:, 5]) #Todoas las filas de la columna 5
Y = np.array(boston.target)

plt.scatter(X, Y, alpha = 0.3)

# Añadimos columna de 1s para termino independiente
X = np.array([np.ones(506), X]).T

B = np.linalg.inv(X.T @ X) @ X.T @ Y

plt.plot([4, 9], [B[0] + B[1]* 4, B[0] + B[1] * 9], c = "red")
plt.show()


# In[17]:


B

