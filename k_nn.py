# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:49:38 2019

@author: georg
"""

import pandas as pandas
import math
import numpy as np

def impute_missing_data(panda_training,panda_testing):
    # This function takes a training and testing panda dataframe, and returns two dataframes with missing values filled in
    
    # the mean and mode selected to add to the dataset come from the combined testing and training files
    
    # get the number of columns in the data array
    data_columns = panda_training.shape[1]
    
    # get a concatenated panda dataframe
    panda_total = pandas.concat([panda_training,panda_testing])
    panda_total_plus = panda_total.where(panda_total[data_columns-1]=='+')
    panda_total_minus = panda_total.where(panda_total[data_columns-1]=='-')
    
    # loop through every column, get the average value, and insert where dataframe is nan
    for i in range(0,data_columns-1):
        
        # remove NaN values from panda_total for the current column
        temp_panda_plus = panda_total_plus.dropna(subset=[i])
        temp_panda_minus = panda_total_minus.dropna(subset=[i])
        
        
        # if the data type is numerical, get the mean, else get the mode
        if panda_total[i].dtype == 'float64':
            column_avg_plus = temp_panda_plus[i].mean()
            column_avg_minus = temp_panda_minus[i].mean()
        elif panda_total[i].dtype == 'int64':
            column_avg_plus = temp_panda_plus[i].mean()
            column_avg_minus = temp_panda_minus[i].mean()
        else:
            column_avg = panda_total[i].mode()
            column_avg = column_avg[0]
            column_avg_plus = column_avg_minus = column_avg
        
        # make the last column in panda training and panda testing equal to the avg values conditioned on plus or minus
        panda_training[data_columns] = np.where(panda_training[data_columns-1]=='+',column_avg_plus,column_avg_minus)
        panda_testing[data_columns] = np.where(panda_testing[data_columns-1]=='+',column_avg_plus,column_avg_minus)
        
        # Insert the average into the column where there is nan, conditioned on + and - in the last column
        panda_training[i] = panda_training[i].fillna(panda_training[data_columns])
        panda_testing[i] = panda_testing[i].fillna(panda_testing[data_columns])
        
    del panda_training[data_columns]
    del panda_testing[data_columns]
    return panda_training, panda_testing;

def normalize_features(panda_training,panda_testing):
    # This function normalizes the real valued-data in the training and testing dataframes
    
    # Normalizing equation is equation 2.1 in the homework set, use the average and stdev from combined data
    
    # get the number of columns in the data array
    data_columns = panda_training.shape[1]
    
    # get the combined data sets
    panda_total = pandas.concat([panda_training,panda_testing])
    
    # loop through every column, get the average value, get the stddev, update value, change column data type to float
    for i in range(0,data_columns-1):
        
        # only change numerical values
        if panda_total[i].dtype == 'float64' or panda_total[i].dtype == 'int64':
            column_avg = panda_total[i].mean()
            column_stdev = panda_total[i].std()
            
            panda_training[i] = panda_training[i]/column_stdev-column_avg/column_stdev
            panda_testing[i] = panda_testing[i]/column_stdev-column_avg/column_stdev
            
            
            
    return panda_training, panda_testing;

def distance(row_1,row_2):
    # this function calculates the distance between two rows
    
    #stores the squared distance
    square_distance = 0
    
    # get the row length
    row_length = row_1.shape[0]
    
    # loop through the row, calculating the distance between each feature
    for i in range(0,row_length-1):
        
        # if row data is numerical, calculate the new distance, else if not numerical and values are not equal add 1 to square distance
        if isinstance(row_1[i],(int,float)):
            square_distance = square_distance + (row_1[i]-row_2[i])*(row_1[i]-row_2[i])
        elif row_1[i] != row_2[i]:
            square_distance = square_distance + 1
            
    l2_distance = math.sqrt(square_distance)
    
    return l2_distance;

def predict(panda_training,panda_testing,k):
    # this function looks at each row of testing data, performs k nearest neighbor analysis on the training data, and makes a prediction for each row
    
        # this function keeps track of distance to all trianing data neighbors and then sorts to find k nearest neighbors, this is much faster than predict 1
    
    # get number of test rows, train rows and columns
    test_size = panda_testing.shape[0]
    train_size = panda_training.shape[0]
    train_columns = panda_training.shape[1]
    
    # initialize a new predictions dataframe
    predictions = pandas.DataFrame(index=np.arange(0,test_size))
    predictions['Predictions'] = 0
    
    # for each test row, add neighbor to array, sort the array, and take the predictions from the k smallest values
    for i in range(0,test_size):
        
        # initialize a k nearest neighbors dataset with high values in distance column
        neighbor_vals = pandas.DataFrame(index=np.arange(0,train_size))
        neighbor_vals[0] = 0
        neighbor_vals[1] = 0
        
        # for every row in the training file, check the distance to the current row, insert into dataframe along with outcome
        for j in range(0,train_size):
            
            # get the distance between the test and training row
            distance_row = distance(panda_testing.iloc[i,:],panda_training.iloc[j,:])
            neighbor_sign = panda_training.iloc[j,train_columns-1]
            
            # add to the neighbor value dataframe
            neighbor_vals.iloc[j,0] = distance_row
            neighbor_vals.iloc[j,1] = neighbor_sign
            
        # order neighbor vals based on distance column
        neighbor_vals = neighbor_vals.sort_values([0])
        
        # take only the k nearest neighbors
        k_nearest = neighbor_vals.iloc[0:k]
        
        # get prediction from mode of k_nearest
        pred = k_nearest[1].mode()
        pred = pred[0]
        
        predictions['Predictions'].iloc[i] = pred

    return predictions;

def accuracy(true_labels,predicted_labels):
    
    # make sure the columns have the same labels
    predicted_labels.columns = list(true_labels)
    
    # get size of column
    column_size = true_labels.shape[0]
    correct_guesses = (true_labels==predicted_labels).sum() 
    
    fraction_correct = correct_guesses.iloc[0]/column_size
    
    return fraction_correct;

crxtestfilepath = 'C:/Users/georg/.spyder-py3/Homework 2/crx.data.testing'
crxtrainfilepath = 'C:/Users/georg/.spyder-py3/Homework 2/crx.data.training'
lensestestfilepath = 'C:/Users/georg/.spyder-py3/Homework 2/lenses.testing'
lensestrainfilepath = 'C:/Users/georg/.spyder-py3/Homework 2/lenses.training'


panda_crx_testing = pandas.read_csv(crxtestfilepath,header=None,na_values='?')
panda_crx_training = pandas.read_csv(crxtrainfilepath,header=None,na_values='?')

panda_lenses_testing = pandas.read_csv(lensestestfilepath,header=None,na_values='?')
panda_lenses_training = pandas.read_csv(lensestrainfilepath,header=None,na_values='?')

# Clean data for crx
panda_crx_training,panda_crx_testing = impute_missing_data(panda_crx_training,panda_crx_testing)
panda_crx_training,panda_crx_testing = normalize_features(panda_crx_training,panda_crx_testing)

# get number of columns crx and lenses
crx_columns = panda_crx_training.shape[1]
lenses_columns = panda_lenses_training.shape[1]

# get predictions for different values of k on both datasets
crx_1k = predict(panda_crx_training,panda_crx_testing,1)
crx_5k = predict(panda_crx_training,panda_crx_testing,5)
crx_9k = predict(panda_crx_training,panda_crx_testing,9)

lenses_1k = predict(panda_lenses_training,panda_lenses_testing,1)
lenses_5k = predict(panda_lenses_training,panda_lenses_testing,5)
lenses_9k = predict(panda_lenses_training,panda_lenses_testing,9)

# Get the accuracy for all predictions
crx_1k_acc = accuracy(panda_crx_testing[crx_columns-1].to_frame(),crx_1k)
crx_5k_acc = accuracy(panda_crx_testing[crx_columns-1].to_frame(),crx_5k)
crx_9k_acc = accuracy(panda_crx_testing[crx_columns-1].to_frame(),crx_9k)

lenses_1k_acc = accuracy(panda_lenses_testing[lenses_columns-1].to_frame(),lenses_1k)
lenses_5k_acc = accuracy(panda_lenses_testing[lenses_columns-1].to_frame(),lenses_5k)
lenses_9k_acc = accuracy(panda_lenses_testing[lenses_columns-1].to_frame(),lenses_9k)


