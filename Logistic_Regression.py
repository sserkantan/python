# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:02:30 2020
Logistic Regression
@author: SSTAN
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = LogisticRegression()
model.fit(x, y)

print("Classes: ", model.classes_)
print("Intercept: ",model.intercept_)
print("Coef: ",model.coef_)
print("Probability: ",model.predict_proba(x))
model.predict(x)
confusion_matrix(y, model.predict(x))


import seaborn as sns

cm = confusion_matrix(y, model.predict(x))
sns.heatmap(cm, annot=True)
print(classification_report(y, model.predict(x)))
model = LogisticRegression(solver='liblinear', C=0.5, random_state=0)
model.fit(x, y)
model.intercept_
model.coef_
model.predict_proba(x)
model.predict(x)
model.score(x, y)
confusion_matrix(y, model.predict(x))
sns.heatmap(confusion_matrix(y, model.predict(x)), annot=True)
print(classification_report(y, model.predict(x)))
