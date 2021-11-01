#!/usr/bin/env python3
# coding=utf-8

import matplotlib.cm  as cm
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from generaDB import Generar

path_file = os.path.dirname(os.path.realpath(__file__))

def LP_datos():
    return Generar()


def data_preprocesados(n, n_components=False):
    dset = [
    "%s/dataset/LP_parametros_1.json" % path_file, 
    "%s/dataset/LP_parametros_2.json" % path_file,
    "%s/dataset/LP_parametros_3.json" % path_file
    ]

    df = pd.read_json(dset[n])
    
    attr_no_stand = [
        # "NroPeaks_th_50", 
        # "NroPeaks_th_75", 
        # "NroPeaks_th_90", 
        "PermEntropy_d5_t1",
        "DetrendedFluctuation",
        "HjorthComplex_d5_t1"
    ]

    X_non_stand = df[attr_no_stand]
    X_stand = df.drop(attr_no_stand + ["BestWavelet", "BestWaveletFq"], axis=1)

    # Encodeamos BestWavelet y BestWaveletFq
    oe = OrdinalEncoder()
    X_oe = oe.fit_transform(df[["BestWavelet", "BestWaveletFq"]])

    # Estandarizamos el resto
    ss = StandardScaler()
    X_ss = ss.fit_transform(X_stand)

    X = np.hstack((X_non_stand.to_numpy(), X_oe, X_ss))

    if isinstance(n_components, int):
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
    
    return X


def plot_silhouette(data, n_clusters, show=True, return_labels=False):
    # REDUCING PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    
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


def plot_DBSCAN_score(outs):

    fig, ax = plt.subplots(1,1)

    max_score = -1
    best_eps = None
    best_min_sample = None
    for out in outs:
        ax.scatter(out[0], out[1], label=f'min_sample: {out[2]}')
        score = max(out[1])
        eps = out[0][np.argmax(out[1])]
        min_sample = out[2]

        if score > max_score:
            max_score = score
            best_eps = eps
            best_min_sample = min_sample

    ax.set_ylim(-0.1, 1)
    ax.set_xlabel('eps')
    ax.set_ylabel('Silouette score')
    ax.legend()
    ax.set_title(f'Best performance:: Score: {max_score:.2f} eps: {best_eps:.1f} min_sample: {best_min_sample}' )

    return fig