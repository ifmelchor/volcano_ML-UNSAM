#!/usr/bin/env python3
# coding=utf-8

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import sys
sys.path.append( '../../../' )
from utils import plot_silhouette, data_preprocesados

X = data_preprocesados(2)

print(' n_cluster  silhouette')
print(' _________  ____________')
for n_clusters in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = km.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'{n_clusters:>6}', f'{silhouette_avg:>10.2f}')
print('')

fig, labels = plot_silhouette(X, 5, return_labels=True, show=False)

file_out = 'kmeans_labels.json'
df = pd.DataFrame({'Index': range(0, len(labels)),'Label': labels}, index=list(df.index), columns=['Index', 'Label'])
df.to_json(file_out)
print(f' database ---> {file_out}')
fig.savefig('silhouette_kmeans.png')

# best fit:
# print('\nNro. clusters: %i' % (2+np.argmax(inertia)))
# print('Silhouette coeficient: %.1f' %max(inertia))

# # ## REDUCING by t-SNE
# tsne = TSNE(n_components=2, perplexity=15.0, init='pca', learning_rate=200)
# dset_embed = tsne.fit_transform(dset_recovered)

# # PLOT
# fig, ax = plt.subplots(1, 1)
# # cmap = 'Spectral_r'
# # z = dset_embed[:,2]
# # norm = mcolor.Normalize(vmin=z.min(), vmax=z.max())
# x, y = dset_embed[:,0], dset_embed[:,1]
# sc = ax.scatter(x, y)#, c=z, cmap=cmap, norm=norm)
# ax.annotate(x)
# ax.set_xlabel('Z1')
# ax.set_ylabel('Z2')
# # fig.colorbar(sc, orientation='vertical', label='Z3')
# plt.show()