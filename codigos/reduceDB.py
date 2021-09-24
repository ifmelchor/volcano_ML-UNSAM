#!/usr/bin/env python3
# coding=utf-8

import matplotlib.cm  as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

## CARGA JSON
dset = 'LP_parametros_3.json' #sys.argv[1]
df = pd.read_json(dset)

## PREPROCESING
# df = df.drop(["Duration"], axis=1)
num_attr = df.drop(["BestWavelet", "BestWaveletFq"], axis=1)
full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), list(num_attr)),
    ("cat", OrdinalEncoder(), ["BestWavelet", "BestWaveletFq"])
])
dset_prepared = full_pipeline.fit_transform(df)

# Matriz correlacion
corr_num = num_attr.corr()
# corr_num["Duration"].sort_values(ascending=False)
fig, ax = plt.subplots(1,1)
sc = ax.imshow(corr_num)
xt = plt.xticks(np.arange(num_attr.shape[1]-1), num_attr.columns[:-1], rotation=45, ha='right', va='top')
yt = plt.yticks(np.arange(num_attr.shape[1]-1), num_attr.columns[:-1], rotation=0, ha='right', va='center')
fig.colorbar(sc, orientation='vertical', label='Correlacion')
plt.show()

## REDUCING by PCA
pca = PCA(n_components=0.9)
dset_reduced = pca.fit_transform(dset_prepared)
dset_recovered = pca.inverse_transform(dset_reduced)
print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())

# ## REDUCING by t-SNE
tsne = TSNE(n_components=2, perplexity=15.0, init='pca', learning_rate=200)
dset_embed = tsne.fit_transform(dset_recovered)

# PLOT
fig, ax = plt.subplots(1, 1)
# cmap = 'Spectral_r'
# z = dset_embed[:,2]
# norm = mcolor.Normalize(vmin=z.min(), vmax=z.max())
x, y = dset_embed[:,0], dset_embed[:,1]
sc = ax.scatter(x, y)#, c=z, cmap=cmap, norm=norm)
ax.annotate(x)
ax.set_xlabel('Z1')
ax.set_ylabel('Z2')
# fig.colorbar(sc, orientation='vertical', label='Z3')
plt.show()