#!/usr/bin/env python3
# coding=utf-8

import matplotlib.cm  as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans

# lee dataset procesado
dset = [
    "../../dataset/LP_parametros_1.json", 
    "../../dataset/LP_parametros_2.json"
    ]
df = pd.read_json(dset[1])

no_stan = [
    "NroPeaks_th_50", 
    "NroPeaks_th_75", 
    "NroPeaks_th_90", 
    "PermEntropy_d5_t1",
    "DetrendedFluctuation",
    "HjorthComplex_d5_t1"
    ] # no se va a estandarizar

# preparando los datos.
X_num1 = df[no_stan]
X_num2 = df.drop(no_stan+["BestWavelet", "BestWaveletFq"], axis=1)

# Encodeamos BestWavelet y BestWaveletFq
oe = OrdinalEncoder()
X_oe = oe.fit_transform(df[["BestWavelet", "BestWaveletFq"]])

# Estandarizamos el resto
ss = StandardScaler()
X_ss = ss.fit_transform(X_num2)

X = np.hstack((X_num1.to_numpy(), X_oe, X_ss))

print(' n_cluster  silhouette')
print(' _________  ____________')
for n_clusters in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = km.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'{n_clusters:>6}', f'{silhouette_avg:>10.2f}')
print('')

def plot_silhouette(data, n_clusters, show=True, return_labels=False):
    # REDUCING PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    # inertia.append(silhouette_score(X, cluster_labels))
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    if show:
        plt.show()

    if return_labels:
        return fig, cluster_labels
    
    else:
        return fig

fig, labels = plot_silhouette(X, 5, return_labels=True, show=False)

file_out = 'kmeans_labels.json'
df = pd.DataFrame({'LP_labels': labels}, index=list(df.index), columns=['LP_labels'])
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