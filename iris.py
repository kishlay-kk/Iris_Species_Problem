# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:40:59 2019

@author: kishl
"""

#Importing the libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv("Iris.csv")
x=dataset.iloc[ :,0:5]
y=dataset.iloc[ :,5:6]
Y=y
#Encoding the y-data  ie. converting it from text to numerical representation

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
y= label.fit_transform(y)
y=np.reshape(y,(150,1))


onehotencode = OneHotEncoder(categorical_features="all")
y=onehotencode.fit_transform(y).toarray()


def convert_to_text (a,b):
    if a[i][0]>0.6 and a[i][1]<0.6 and a[i][2]<0.6 :
      b.append("Iris-setosa")
    elif a[i][0]<0.6 and a[i][1]>0.6 and a[i][2]<0.6 :
      b.append("Iris-versicolor")
    else:
      b.append("Iris-virginica")

#Splitting into test and training sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

# Fitting the regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
regressor.fit(x_train,y_train)

# Predicting the values
y_pred = regressor.predict(x_test)

#Comparing the results

predicted_list = list()
actual_list=list()
for i in range(45):
    convert_to_text(y_pred,predicted_list)

for i in range(45):
    convert_to_text(y_test,actual_list)
count=0
for i in range(45):
    if predicted_list[i]==actual_list[i]:
        count=count+1

Accuracy=(count/45)*100
print("Accuracy of the predictor is:",round(Accuracy,2),"%")


        
    