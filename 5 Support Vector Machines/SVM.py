#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# In[ ]:


# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# TODO: Assign the features to the variable X, and the labels to the variable y. 
X = data[:, :2]
y = data[:, 2]


# In[ ]:


# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel="rbf", gamma=27)

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
train_acc = accuracy_score(y, y_pred)
print(f"{train_acc=}")
