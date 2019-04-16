#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:43:11 2019

@author: bhai
"""

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("knngraph.csv")
x = data.values[:,0:2]
y = data.values[:,2]
print(x)
print(y)
from sklearn.neighbors import KNeighborsClassifier
nei = KNeighborsClassifier(n_neighbors=3,weights="distance")
nei.fit(x,y)
print("THE result is for point [5,3] is")
print(nei.predict([[5,3]]))
print("THE result is for point [6,6] is")
print(nei.predict([[6,6]]))
print("THE result is for point [6,2] is")
print(nei.predict([[6,2]]))
plt.figure()
plt.scatter(x[:, 0], x[:, 1],color="r",marker="o",s=30)
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())
plt.title("Data points")
plt.show()