#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:19:08 2021

@author: francisco
"""

# https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f

import sklearn.datasets as ds
import numpy as np
seed = 42
np.random.seed(seed)

n_samples = 100
n_features = 6
n_informative = np.random.randint(low = 1, high = n_features)
n_redundant = np.random.randint(low = 1, high = n_features-n_informative)
n_repeated = 0
n_classes = 3
n_clusters_per_class = 1
weights = [0.7, 0.2]
flip_y = 0.05

random_state = 42
shuffle = True
X, y = ds.make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative = n_informative,
    n_redundant = n_redundant,
    n_repeated = n_repeated,
    n_classes = n_classes,
    n_clusters_per_class = n_clusters_per_class,
    weights = weights,
    flip_y = flip_y,
    random_state=seed)

