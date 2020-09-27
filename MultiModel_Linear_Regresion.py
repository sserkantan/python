# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:18:51 2020
MultiModel_Linear_Regresion
@author: SSTAN
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
X,y=make_regression(n_samples=100, n_features=3,noise=5)
import matplotlib.pyplot as plt
m_model=LinearRegression()
m_model.fit(X,y)
X_test,y_test=make_regression(n_samples=100, n_features=3,noise=5)
print(m_model.coef_)
print(m_model.intercept_)
y_predict=m_model.predict(X_test)
#plt.scatter(y_test,y_pred)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix((y_test,y_pred,Labels=None))


