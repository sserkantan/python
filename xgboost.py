# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:47:27 2020
xgboost
@author: SSTAN
"""

import pandas as pd
import seaborn as sns
import numpy as np
data = pd.read_csv("iris.csv")
print(data.head())
print("\n")
print(data.describe())
print("\n")
print(data.info())
print("\n")
print(data.groupby(by="Species").mean())
#print(data.groupby(by="Species").agg(["mean","min","max")))
print("\n")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

label_encoder = LabelEncoder()
data["Species"] = label_encoder.fit_transform(data["Species"])
    
print(data.head())
print(data["Species"].value_counts())
data.drop("Id", axis=1, inplace=True)
print(data.head())
X, y = data.iloc[:, :-1], data.iloc[:, -1]
#X, y = data.iloc[0:,5:], data.iloc["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

import xgboost as xgb


dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)
param = {'max_depth':3, 
         'eta':1, 
         'objective':'multi:softprob', 
         'num_class':3}
num_round = 5
model = xgb.train(param, dmatrix_train, num_round)
preds = model.predict(dmatrix_test)
preds[:10]
best_preds = np.asarray([np.argmax(line) for line in preds])
print(best_preds)
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, square=True, annot=True, cbar=False)