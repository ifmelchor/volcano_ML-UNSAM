#!/usr/bin/env python3
# coding=utf-8

import matplotlib.cm  as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

## CARGA JSON
dset = sys.argv[1]
df = pd.read_json(dset)

## PREPROCESING
num_attr = df.drop(["BestWavelet", "BestWaveletFq"], axis=1)
full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), list(num_attr)),
    ("cat", OneHotEncoder(), ["BestWavelet", "BestWaveletFq"])
])
dset_prepared = full_pipeline.fit_transform(df)

# ## REDUCING by PCA
# pca = PCA(n_components=3) # continue el 75% de la variancia de los datos
# dset_reduced = pca.fit_transform(dset_prepared)
# dset_recovered = pca.inverse_transform(dset_reduced)
# print(pca.explained_variance_ratio_)

## REDUCING by t-SNE
tsne = TSNE(n_components=3, perplexity=10.0, init='pca')
dset_embed = tsne.fit_transform(dset_prepared)

# PLOT
fig, ax = plt.subplots(1, 1)
cmap = 'Spectral_r'
z = dset_embed[:,2]
norm = mcolor.Normalize(vmin=z.min(), vmax=z.max())
x, y = dset_embed[:,0], dset_embed[:,1]
sc = ax.scatter(x, y, c=z, cmap=cmap, norm=norm)
ax.set_xlabel('Z1')
ax.set_ylabel('Z2')
fig.colorbar(sc, orientation='vertical', label='Z3')
plt.show()