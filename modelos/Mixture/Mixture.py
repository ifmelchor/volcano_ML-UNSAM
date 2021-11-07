#!/usr/bin/env python3
# coding=utf-8

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

import sys
sys.path.append( '../..' )
from utils import data_preprocesados, plot_mixture

X = data_preprocesados(2, n_components=2, include_categorical=False)

k = 4
gmm = BayesianGaussianMixture(n_components=k, n_init=10).fit(X)
plot_mixture(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

prob = gmm.predict_proba(X)

# buscamos los más probables y hacemos plots comparativos para definir etiuetas y pasar al semisupervisado
