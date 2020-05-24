# %%
import sys
import numpy as np 
import os
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras

# %%
iris = load_iris()
X = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, .5]])

# %%
y_pred 

# %%
tf.__version__

# %%
keras.__version__

# %%
