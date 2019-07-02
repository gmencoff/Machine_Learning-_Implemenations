# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 00:32:21 2019

@author: georg
"""

import numpy as np
import scipy as scipy
from scipy import io
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix

data = scipy.io.loadmat('C:/Users/georg/.spyder-py3/Homework 4/MNIST_data.mat')

def SVM_Model(x,t,C,kernel):
    # this function accepts training data, C and kernal values as an input and outputs a model in a panda dataframe
    # for both 1vs 1 and 1vs. rest classification schemes
    
    classes = np.unique(t) # get unique classes
    num_classes = len(classes) # get the number of classes
    
    num_models = int(num_classes + num_classes*(num_classes-1)/2) # calculate the number of models which is the number of classes (1 vs rest) + 1vs1
    
    oneVone_pairs = oneVone(num_classes) # get the data pairs for 1v1
    
    models = pd.DataFrame(index=['a','x','t','b'],columns=[])# initialize the models dataframe
    
    b = [] # store b values
    
    # loop through calculating all possible models
    for iModel in range(0,num_models):
        
        # check if the model is a 1vall or 1v1, and get data
        if iModel < num_classes:
            x_scheme,t_scheme,model_name = voteScheme(x,t,num_classes,num_models,iModel,oneVone_pairs)
        else:
            x_scheme,t_scheme,model_name = voteScheme(x,t,num_classes,num_models,iModel,oneVone_pairs)
            
        # get the length of the data
        a_length = len(t_scheme)
        
        # get the T matrix
        T = np.diag(t_scheme)
        
        # get the K matrix
        K = generateKernal(x_scheme,kernel)
        
        # Define the parameters for input into the optimizer, and call the optimizer
        P = T*K*T
        q = np.array([[-1]*a_length]).T
        G = np.append([[1]*a_length],[[-1]*a_length],axis=0)
        h = np.array([[C],[0]])
        A = np.array([t_scheme])
        b = np.array([[0]])
        
        a = solvers.qp(matrix(P.astype('double')),matrix(q.astype('double')),matrix(G.astype('double')),matrix(h.astype('double')),matrix(A.astype('double')),matrix(b.astype('double')))
        
        # add b to the model
        b_temp = calcb(a['x'],t_scheme,K,C)
        models[model_name] = [np.array(a['x']),x_scheme,t_scheme,b_temp]
    
    return models

def voteScheme(x,t,num_classes,num_models,iModel,oneVone_pairs):
    data_num = len(t)
    t_scheme = []
    x_scheme = []
    
    if iModel < num_classes:
        # create a 1 vs rest dataset
        x_scheme = x
        model_name = str(iModel) + ' vs Rest'
        
        for i in range(0,data_num):
            if t[i,0] == iModel:
                t_scheme.append(1)
            else:
                t_scheme.append(-1)
        
    else:
        #create a 1 vs 1 dataset
        class_1 = oneVone_pairs[0,(iModel-num_models)]
        class_2 = oneVone_pairs[1,(iModel-num_models)]
        
        model_name = str(class_1) + ' vs ' + str(class_2)
        
        for i in range(0,data_num):
            if t[i,0] == class_1:
                if x_scheme == []:
                    x_scheme = [x[i,:]]
                else:
                    x_scheme = np.concatenate((x_scheme,[x[i,:]]),0)
                t_scheme.append(1)
            elif t[i,0] == class_2:
                if x_scheme == []:
                    x_scheme = np.array([x[i,:]])
                else:
                    x_scheme = np.concatenate((x_scheme,[x[i,:]]),0)
                t_scheme.append(-1)
                           
    
    return x_scheme,t_scheme,model_name

def oneVone(class_num):
    index_1 = 0
    index_2 = 2
    index_2_start = 1
    model_pairs = np.array([[index_1],[index_2_start]])
    
    pair_num = int(class_num*(class_num-1)/2)
    
    for iPair in range(1,pair_num):
        # check if index 2 exceeds array bounds and reset indices to new location
        if index_2%class_num == 0:
            index_1 = index_2_start
            index_2 = index_2_start+1
            index_2_start = index_2_start+1
       
        model_pairs = np.concatenate((model_pairs,np.array([[index_1],[index_2]])),1)
        
        index_2 = index_2+1        
    
    return model_pairs

def generateKernal(x,kernel):
    # this function generates a kernal matrix for either a gaussian or polynomial kernal
    
    # to specify polynomial, use first letter p, second letter order
    # to speficy gaussian, use letter g    
    # specify the variance
    
    data_points,data_features = np.shape(x)
    K = np.array([[0 for i in range(data_points)] for j in range(data_points)])
    
    for i in range(data_points):
        for j in range(data_points):
            if kernel[0] == 'p':
                order = int(kernel[1])
                K[i][j] = polyKernal(x[i,:],x[j,:],order)
            else:
                K[i][j] = rbfKernal(x[i,:],x[j,:],1)
    
    return K

def polyKernal(x1,x2,order):
    
    k = (1+np.dot(x1,x2))**order
    
    return k

def rbfKernal(x1,x2,var):
    
    k = np.exp(-np.linalg.norm(x1-x2)/(2*var**2))
    
    return k

def calcb(a,t_scheme,K,C):
    # calculate b for each model
    num_data = len(t_scheme)
    a = np.array(a)
    b = 0
    counta = 0
    
    for i in range(0,num_data):
        an = a[i]
        if an > 0 and an < C:
            counta = counta+1
            jsum = 0
            for j in range(0,num_data):
                jsum = jsum + a[j]*t_scheme[j]*K[i][j] 
        b = b + (t_scheme[i]-jsum)            
    
    return b/counta

def predictModel(model,test_samples,kernel):
    # for one vs all, see which gives the largest positive or negative value
    rows,model_num = model.shape
    num_samples,features = test_samples.shape
    
    class_num = int((-1+np.sqrt(1+8*model_num))/2)
    
    # initialize the overall predictions
    t_predicted_one_rest = np.zeros(num_samples)
    t_predicted_one_one = np.zeros(num_samples)
    
    # create a prediction for each test sample
    for iSample in range(0,num_samples):
        # get test point
        test_x = test_samples[iSample,:]
        
        # initialize one vs rest and one vs one predictions
        one_rest_predictions = np.zeros(class_num)
        one_one_predictions = np.zeros(class_num)
        
        # predict based on one vs rest
        for iOneRest in range(0,class_num): 
            # get the data from the model
            model_name = str(iOneRest) + ' vs Rest'
            a = model[model_name]['a']
            x = model[model_name]['x']
            t = model[model_name]['t']
            b = model[model_name]['b']
            one_rest_predictions[iOneRest] = predictPoint(a,t,x,b,kernel,test_x)
        
        predicted_point = np.argmax(one_rest_predictions)
        t_predicted_one_rest[iSample] = predicted_point
        
        class_1 = 0
        class_2_start = 1
        class_2 = 1
        
        # make a prediction using one vs one
        for iOneOne in range(0,(model_num-class_num)):
            model_name = str(class_1) + ' vs ' + str(class_2)
            a = model[model_name]['a']
            x = model[model_name]['x']
            t = model[model_name]['t']
            b = model[model_name]['b']
            y = predictPoint(a,t,x,b,kernel,test_x)
            
            # whicher point wins, add one to the counter
            if y > 0:
                one_one_predictions[class_1] = one_one_predictions[class_1] + 1
            else:
                one_one_predictions[class_2] = one_one_predictions[class_2] + 1
            
            class_2 = class_2 + 1
            
            if class_2%class_num == 0:
                class_1 = class_2_start
                class_2 = class_2_start + 1
                class_2_start = class_2
                
        predicted_point = np.argmax(one_one_predictions)
        t_predicted_one_one[iSample] = predicted_point            
        
    return t_predicted_one_rest,t_predicted_one_one

def predictPoint(a,t,x,b,kernel,test_x):
    num_points = len(a)
    y = 0
    
    if kernel[0] == 'p':
        # polynomial kernal
        order = int(kernel[1])
        for i in range(1,num_points): 
            y = y+a[i]*t[i]*polyKernal(x[i,:],test_x,order)
    else:
        # gaussian kernal
        for i in range(1,num_points): 
            y = y+a[i]*t[i]*rbfKernal(x[i,:],test_x,1)
    
    return y+b

kernel = 'p3'
x = data['train_samples']
t = data['train_samples_labels']
t_test = data['test_samples']
t_test_labels = data['test_samples_labels']
model = SVM_Model(x[0:100,:],t[0:100],.1,kernel)
t_predicted_one_rest,t_predicted_one_one = predictModel(model,data['test_samples'],kernel)

confusion_matrix_one_rest = confusion_matrix(t_test_labels,t_predicted_one_rest)
confusion_matrix_one_one = confusion_matrix(t_test_labels,t_predicted_one_one)

one_rest_accuracy = np.trace(confusion_matrix_one_rest)/sum(sum(confusion_matrix_one_rest))
one_one_accuracy = np.trace(confusion_matrix_one_one)/sum(sum(confusion_matrix_one_one))
