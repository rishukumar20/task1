#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf  #this is used to import tensorflow for model creation and updating
import pandas as pd    #used for data cleaning analysis
import numpy as np     #used for array computing


# In[2]:


import keras  #high level neural network


# In[3]:


import glob  #used to return all file paths that match a specific pattern


# In[4]:


train_files=glob.glob("D:/brain tumour/*/**")


# In[5]:


train_files #gives location of all files


# In[8]:


train_files[0] #return location of first file in dataset


# In[10]:


import matplotlib.pyplot as plt #used for plotting graph
from PIL import Image           # help in loading and manipulating image


# In[17]:


img1=Image.open(train_files[2])  #return image at corresponding location


# In[18]:


img1


# In[19]:


img1_array=np.array(img1)  #numphy used here to convert img1 into 3-D array


# In[20]:


print( img1_array.shape)   ##gives order layers,rows and columns of the defined dataset
print(img1_array)

