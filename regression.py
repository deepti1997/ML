#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:45:43 2019

@author: ccoew
"""


import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b,x_t=0,y_pred_val=0): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
    if x_t!=0 and y_pred_val!=0:
        plt.plot(x_t, y_pred_val,"ro") 

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([10, 9, 2, 15, 10, 16, 11, 16]) 
    y = np.array([95, 80, 10, 50, 45, 98, 38, 83]) 
    #x = np.array([1, 2, 4, 3, 5]) 
    #y = np.array([1, 3, 3, 2, 5])
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\n b_0 = {}  \n b_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
    
    #predict value for X=8
    print("Enter X value : ")
    x_t = int(input())
#    x_t = 8
    y_pred = b[0]+(b[1]*x_t)
    print("Predicted value for x = {} is y = {} ",x_t,y_pred)  
    #Final Plot
    plot_regression_line(x, y, b,x_t,y_pred)
if __name__ == "__main__": 
    main() 
