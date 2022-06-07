"""
Notebook inspired by [scikit-learn example](https://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html), but
implemented with tf.keras
"""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces

#%% Load the faces datasets
data, targets = fetch_olivetti_faces(return_X_y=True)
data = data.reshape(-1, 64, 64)

#%% Some stat on dataset
Counter(targets)
pd.Series(targets).value_counts().sort_index()

#%% Visualize some data

#%% Create train and test sets

train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

n_pixels = data.shape[1]
# Upper half of the faces
X_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
