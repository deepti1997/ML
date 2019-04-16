#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:50:56 2019

@author: ccoew
"""

import sklearn
import numpy as np
import pandas as pd

buy = pd.read_csv('buys.csv')

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

x_fe=['age','income','gender','marital_status']
x=buy[x_fe]
y=buy['buys']

from sklearn import preprocessing
x=x.apply(preprocessing.LabelEncoder().fit_transform)

#encoding the categories 
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()
"""


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(x,y)


from IPython.display import Image  
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(classifier, out_file=None, 
                                feature_names=x_fe,class_names=['no', 'yes'], filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
graph.write_png("dtreenew.png")
