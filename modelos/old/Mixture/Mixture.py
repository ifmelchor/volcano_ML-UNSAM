#!/usr/bin/env python3
# coding=utf-8

import pandas as pd
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

import sys
sys.path.append( '../..' )
from utils import data_preprocesados, plot_mixture, LP_datos

X = data_preprocesados(2, n_components=2, include_categorical=False)

k = 4
gmm = GaussianMixture(n_components=k, n_init=10).fit(X)
labels = gmm.predict(X)
prob = gmm.predict_proba(X)

plot_mixture(X, labels, gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

index, label = np.where(prob>0.9)
str_index = [r'LP$_{%i}$' % i for i in index]
file_out = 'GMM.json'
df = pd.DataFrame({'Index': index, 'Label': label}, index=index, columns=['Index', 'Label'])
df.to_json(file_out)
print(f' database ---> {file_out}')