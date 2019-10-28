# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:45:05 2019

@author: Gautam sachdev
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

df = pd.read_csv("50_startups.csv");
x = df.iloc[0:50,0:4].values;
y = df.iloc[0:50,4].values;

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#dummy variable trap
x = x[:,1:6];

#fitting Ml model to regression;
from sklearn.linear_model import LinearRegression;
reg = LinearRegression();
reg.fit(x,y);

#visualizing data results
prediction = reg.predict(x);
