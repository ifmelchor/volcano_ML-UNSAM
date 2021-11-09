#!/usr/bin/env python3
# coding=utf-8

import matplotlib as mpl
import matplotlib.cm  as cm
import matplotlib.colors  as mcolor
import matplotlib.pyplot as plt

import os
import itertools
import pandas as pd
import numpy as np
from collections import Counter
from scipy import linalg

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from generaDB import Generar
import scicomap as sc

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

path_file = os.path.dirname(os.path.realpath(__file__))

def LP_datos():
    return Generar()


def data_preprocesados(n, onehot=False, n_components=False, include_categorical=True):
    dset = [
    "%s/dataset/LP_parametros_1.json" % path_file, 
    "%s/dataset/LP_parametros_2.json" % path_file,
    "%s/dataset/LP_parametros_3.json" % path_file
    ]

    df = pd.read_json(dset[n])
    
    attr_no_stand = [
        "PermEntropy_d5_t1",
        "DetrendedFluctuation",
        "HjorthComplex_d5_t1"
    ]

    X_non_stand = df[attr_no_stand]
    X_stand = df.drop(attr_no_stand + ["BestWavelet", "BestWaveletFq"], axis=1)

    # Encodeamos BestWavelet y BestWaveletFq
    if onehot:
        oe = OneHotEncoder()
        X_oe = oe.fit_transform(df[["BestWavelet", "BestWaveletFq"]]).toarray()
    else:
        oe = OrdinalEncoder()
        X_oe = oe.fit_transform(df[["BestWavelet", "BestWaveletFq"]])


    # Estandarizamos el resto
    ss = StandardScaler()
    X_ss = ss.fit_transform(X_stand)

    if include_categorical:
        X = np.hstack((X_non_stand.to_numpy(), X_oe, X_ss))
    else:
        X = np.hstack((X_non_stand.to_numpy(), X_ss))

    if isinstance(n_components, int):
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
    
    return X


def plot_labels(X, labels, show=True):

    # compute score and n_cluster
    score = silhouette_score(X, labels)

    # Reduce for plot
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(f'Best score: {score:.2f}')

    sc.ScicoQualitative(cmap='')

    sc_map = cm.get_cmap('jet')
    norm = mcolor.Normalize(
        vmin=min(Counter(labels).keys()), 
        vmax=max(Counter(labels).keys())
        )
    ax1.scatter(
        X_reduced[:, 0], 
        X_reduced[:, 1], 
        marker='o', 
        s=30, 
        lw=0, 
        alpha=0.7,
        c=sc_map(norm(labels)),
        edgecolor='k'
        )
    ax1.set_xlabel(r'X$_1$')
    ax1.set_ylabel(r'X$_2$')
    
    handles = []
    for n, l in enumerate(Counter(labels).keys()):
        label = str(l)
        color = sc_map(norm(l))
        h = [plt.Line2D([0], [0], linestyle="none", marker="o", color=color, alpha=1)]
        handles += [(n, h, label)]

    ax1.legend([h[1][0] for h in handles], [h[2] for h in handles], title='Labels')

    plt.show()
    return fig


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


def plot_mixture(X, Y_, means, covariances, index, title):

    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()


def plot_LP_list(lp_list, show=True):
    fig, ax = plt.subplots(1,2, figsize=(16,4), gridspec_kw={'width_ratios':[1,0.5]})
    x = 0
    for n, lp in enumerate(lp_list):
        ax[0].plot(lp.time, lp.data/lp.data.max()+n*2)
        if lp.time[-1]>x:
            x = lp.time[-1]
        freq, PSD = lp.get_psd((1,6), normalize=True)
        ax[1].plot(freq, PSD, label=r'LP$_{%i}$' %lp.index)
    ax[0].set_xlim(0, x)
    ax[0].set_xlabel('Tiempo [sec]')
    ax[0].set_yticks([])
    ax[1].set_xlabel('Frecuencia [Hz]')
    ax[1].set_ylabel('PSD')

    fig.legend()

    if show:
        plt.show()
        plt.close()
    
    else:
        return fig


def combine_list(ldl):
    it = True

    while it:
        for xi, xj in itertools.product(ldl,ldl):
            if xi != xj:
                if list(set(xi) & set(xj)):
                    i = ldl.index(xi)
                    j = ldl.index(xj)
                    new_item = list(set(xi + xj))
                    ldl[i] = new_item
                    ldl.remove(ldl[j])

                    if len(ldl) > 1:
                        it = True
                        break
                    else:
                        it = False
                
                else:
                    it = False
    
    return ldl