# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:30:53 2019

@author: georg
"""

import numpy as np
import scipy as scipy
from scipy import io
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def denoiseImg(img,h,B,n,maxits):
    # intiialize cleaned image to the original
    cleaned_img = img
    
    # get the size of the image
    rows,columns = img.shape
    
    pixel_flipped = 1
    its = 0
    
    #loop through each pixel
    while pixel_flipped == 1 and its < maxits:
        pixel_flipped = 0
        for iRow in range(0,rows):
            for iColumn in range(0,columns):
                x_pixel = cleaned_img[iRow,iColumn]
                y_pixel = img[iRow,iColumn]
                
                # get neighbor values
                neighbors = []
                neighbor_vals = []
                
                if iRow+1<rows:
                    neighbors.append([iRow+1,iColumn])
                if iRow-1>=0:
                    neighbors.append([iRow-1,iColumn])
                if iColumn+1<columns:
                    neighbors.append([iRow,iColumn+1])
                if iColumn-1>=0:
                    neighbors.append([iRow,iColumn-1])
                
                for i in range(0,len(neighbors)):
                    neighbor_idx = neighbors[i]
                    neighbor_val = cleaned_img[neighbor_idx[0]][neighbor_idx[1]]
                    neighbor_vals.append(pixelConvert(neighbor_val))
                    
                # calculate pixel energy
                pixel_value = pixelEnergy(pixelConvert(x_pixel),pixelConvert(y_pixel),neighbor_vals,h,B,n)
                pixel_value_flip = pixelEnergy(-pixelConvert(x_pixel),pixelConvert(y_pixel),neighbor_vals,h,B,n)
                
                # insert lower energy pixel
                if pixel_value < pixel_value_flip:
                    cleaned_img[iRow,iColumn] = x_pixel
                else:
                    flippedpixel = pixelConvert(-pixelConvert(x_pixel))
                    cleaned_img[iRow,iColumn] = flippedpixel
                    # record that a pixel was flipped
                    pixel_flipped = 1
        its = its+1
    
    return cleaned_img

def pixelEnergy(pixel,ypixel,neighbor_vals,h,B,n):
    # calculate pixel energy
    pix_energy = h*pixel-n*pixel*ypixel
    
    for i in range(0,len(neighbor_vals)):
        pix_energy = pix_energy-B*pixel*neighbor_vals[i]
    
    return pix_energy

def pixelConvert(pixel):
    # convert from 200 to 1, 25 to -1 and vice versa
    if pixel==200:
        pixel=1
    elif pixel==25:
        pixel=-1
    elif pixel==1:
        pixel=200
    elif pixel==-1:
        pixel=25
    
    return pixel



img_noisy = scipy.misc.imread('C:/Users/georg/.spyder-py3/Homework 4/Bayes_noisy.png')

img_cleaned = denoiseImg(img_noisy,0,1,2,100)
imgplot = plt.imshow(img_cleaned)
plt.show()
