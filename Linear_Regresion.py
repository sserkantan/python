    # -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:28:51 2020
Linear_Regresiion
@author: SSTAN
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
X,y=make_regression(n_samples=100, n_features=1,noise=5)
import matplotlib.pyplot as plt
plt.scatter(X,y)
model=LinearRegression()
model.fit(X,y)
model.score(X,y)
model.coef_
model.intercept_
X_test,_=make_regression(n_samples=100, n_features=1,noise=5)
y_pred=model.predict(X_test)
plt.scatter(X,y, color="red")
plt.scatter(X_test,y_pred)

