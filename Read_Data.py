# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:10:31 2020
Read Data

@author: SSTAN
"""

import pandas as pd
import seaborn as sns
import numpy as np
data = pd.read_csv("iris.csv")
print(data.head())
print(data.describe())
print(data.info())
print(data.groupby(by="Species").count())
print(data.groupby(by="Species").mean())
print(data.groupby(by="Species").max())
print(data.groupby(by="Species").min())
sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=data)
sns.pairplot(data, hue="Species")
