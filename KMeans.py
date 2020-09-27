# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:04:52 2020
KMeans 
@author: SSTAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, n_features=2, cluster_std=5.5, random_state=42)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sns.set_style("whitegrid")
sns.scatterplot(X_train.T[0], X_train.T[1], hue=y_train)
sns.scatterplot(X_test.T[0], X_test.T[1], hue=y_test)


from sklearn.cluster import KMeans

model = KMeans(n_clusters=1)
var=[]
for n in range(1, 15):
    model = KMeans(n_clusters=n)
    model.fit(X_train)
    var.append(model.inertia_)
    print(model.inertia_)
    
plt.figure(figsize=(10,5))
#<plt.set_style("whitegrid")
plt.plot(var)
plt.xticks(range(1,15))
plt.show()
