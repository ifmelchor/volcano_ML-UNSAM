#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import sys
sys.path.append( '../..' )
from utils import data_preprocesados, plot_DBSCAN_score

X = data_preprocesados(2, n_components=2)

# outs = []
# for s in [10,15,25,50]:
#     eps = []
#     scores = []
#     for e in np.arange(2,6.5,0.05):
#         dbscan = DBSCAN(eps=e, min_samples=s)
#         dbscan.fit(X)
#         if len(Counter(dbscan.labels_)) > 1:
#             score = silhouette_score(X, dbscan.labels_)
#         else:
#             score = -1
#         eps.append(e)
#         scores.append(score)
#     outs.append((eps, scores, s))

# fig = plot_DBSCAN_score(outs)
# plt.show()

# mejor silohuette score:
eps = 5
min_samples = 15
dbest = DBSCAN(eps=eps, min_samples=min_samples)
dbest.fit(X)

print(Counter(dbest.labels_))
# El mejor resultado no ofrece grandes soluciones.
# DBSCAN no es capaz de separar entre clusters.

print(np.where(dbest.labels_<0))
# LP_anomalos: 206, 211, 365, 369, 446, 627, 945