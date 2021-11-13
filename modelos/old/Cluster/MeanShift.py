#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift

import sys
sys.path.append( '../..' )
from utils import data_preprocesados, plot_labels

X = data_preprocesados(2, onehot=False, n_components=None, include_categorical=True)

ms = MeanShift(cluster_all=False).fit(X)
print('Num. labels:', len(Counter(ms.labels_)))
print('Score:', silhouette_score(X, ms.labels_))
print(Counter(ms.labels_))

plot_labels(X, ms.labels_)