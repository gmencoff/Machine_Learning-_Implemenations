# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:14:29 2019

@author: georg
"""

import numpy as np

def build_design_matrix(x):
    # this function builds a design matrix from an x array of inputs
    # this function always takes 3 data arrays as an input and returns the design matrix
    # this function builds a design matrix for a linear combination of data inputs
    
    # get the length of the data
    data_size = x.shape[1]
    
    # get width of data
    data_width = x.shape[0]
    
    # initialize the design matrix
    design_matrix = np.matrix([[0 for i in range(data_size+1)] for j in range(data_width)])
    
    # populate the design matrix
    for i in range(data_width):
        
        design_matrix[i,0] = 1
        
        for j in range(data_size):
            design_matrix[i,j+1] = x[i,j]
            
    return design_matrix;

def get_weights(design_matrix,t):
    # this function takes the design matrix and the pbserved t data values, and outputs an array of weights
    
    return ((design_matrix.T*design_matrix).I)*design_matrix.T*t

def sum_square_error(w,t,design_matrix):
    # this function returns the sum of the square errors using the given weights and the given design matrix
    
    tn_minus_dw = t-design_matrix*w
    
    return tn_minus_dw.T*tn_minus_dw;
    
            
    
filepath = 'C:/Users/georg/.spyder-py3/Homework 2/detroit.npy'
hom_data = np.load(filepath)
file_dict = {0:'FTP',1:'UEMP',2:'MAN',3:'LIC',4:'GR',5:'NMAN',6:'GOV',7:'HE',8:'WE',9:'HOM'}

x_total = np.matrix([hom_data[:,0],hom_data[:,1],hom_data[:,2],hom_data[:,3],hom_data[:,4],hom_data[:,5],hom_data[:,6],hom_data[:,7],hom_data[:,8]]).T
t = np.matrix(hom_data[:,9]).T

input_1_idx = 0
input_2_idx = 8
number_inputs_to_test = 7
results = []
results_simple = []


for i in range(number_inputs_to_test):
    x = np.matrix([hom_data[:,input_1_idx],hom_data[:,input_2_idx],hom_data[:,i+1]]).T
    design = build_design_matrix(x)
    weights = get_weights(design,t)
    error = sum_square_error(weights,t,design)
    
    results.append([file_dict[i+1],error])
    
    
