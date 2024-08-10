#!/usr/bin/env python
# coding: utf-8

# # Neuronal connectivity based on cell morphologies
# ## KTH: Project course in Scientific Computing 


import os 

#Importing the basic necessary packages and libraries
import matplotlib.pyplot as plt
from   matplotlib import cm
from scipy.stats import skewnorm
import seaborn as sns
import numpy as np
import collections


from   scipy.stats import (multivariate_normal as mvn, norm)
from   scipy.stats._multivariate import _squeeze_output
from   matplotlib import cm

from   scipy._lib._util import check_random_state
from   scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen
from   scipy.special import gammaln

#Initiate a space of 100 neurons randomly as a matrix with entries zeros and ones
#Create a matrix of shape (200,200) with 100 random entries being one and the rest zero
#The ones represent the position of somata

nums = np.zeros((200,200))                #Matrix of shape (200,200) with entries 0 
nums[:1,0:100] = 1                        #100 entries should be one
nums = nums.flatten()                     #flatten matrix for shuffling
np.random.shuffle(nums)                   #shuffle entries 
nums = np.reshape(nums, (200,200))        #Reshape into matrix of shape (200,200)

nums_pad = np.pad(nums,100)               #Add padding for future convolution, nums_pad has shape (400,400)
# get_ipython().run_line_magic('store', 'nums')
# get_ipython().run_line_magic('store', 'nums_pad')


def soma_locations(neuron_space): 
    
    """
    soma_locations infers the position of all somata in the space of all neurons

    :param neuron_space: 2D array of arbitrary length 
    :return: 2D array containing the spacial location of one entries in neuron_space
    """
    
    locations = []
    for i in range(0,neuron_space.shape[0]):
        for j in range(0,neuron_space.shape[1]):
            if neuron_space[i][j] == 1: 
                locations.append(np.array([i,j]))
    return np.array(locations)


def connectivity_matrix(axon_morph, dendrite_morph, thr):
    
    """
    connectivity_matrix creates a connectivty matrix via convolution given a threshold value

    :param axon_morph: 2D array of arbitrary length that specifies the shape of the axon, i.e. could be density matrix
    :param dendrite_morph: 2D array of arbitrary length that specifies the shape of the dendrite, i.e. could be density matrix
    :thr: floating point number
    
    :return: 2D array of shape (100,100) specifying the neuronal connectivity
    """
    
    D = np.empty((100,100))
    for x in range(0,100):
        for y in range(0,100):
            B = np.zeros((400,400))
            C = np.zeros((400,400))
            r1 = soma_locations(nums_pad)[x][0]-50
            c1 = soma_locations(nums_pad)[x][1]-50
            r2 = soma_locations(nums_pad)[y][0]-50
            c2 = soma_locations(nums_pad)[y][1]-50
        
            # B[r1:r1+axon_morph.shape[0], c1:c1+axon_morph.shape[1]] = axon_morph --> positions the upper left corner of axon_morph at position (r1,c1) of matrix B
            B[r1:r1+axon_morph.shape[0], c1:c1+axon_morph.shape[1]] = axon_morph 
            
            # C[r2:r2+dendrite_morph.shape[0], c2:c2+dendrite_morph.shape[1]] --> positions the upper left corner of dendrite_morph at position (r2,c2) of matrix C
            C[r2:r2+dendrite_morph.shape[0], c2:c2+dendrite_morph.shape[1]] = dendrite_morph
            
            if sum(sum(np.multiply(B,C))) >= thr: 
                D[x][y]=1            #entry=1 if axon of neuron x connects to dedrite of neuron y
            else: 
                D[x][y]=0
                
    return D

ss = np.load('shape_density_matrix.npz')
sp = ss['shapes']
connectivity_matrices_thr1 = []

nn=0
for ii in range(6):
    nn=nn+1
    print(nn)
    x_shape = sp[0]
    y_shape = sp[ii]
    connectivity_matrices_thr1.append(connectivity_matrix(x_shape, y_shape, 2e-6))

np.savez('conn_mat_shape_1.npz', conn_mat = connectivity_matrices_thr1)
