#!/usr/bin/env python3
# coding=utf-8


import matplotlib.pyplot as plt
import os
import csv
import itertools
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA

from generaDB import Generar

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

path_file = os.path.dirname(os.path.realpath(__file__))

def LP_datos():
    return Generar()


def LP_PSDs(test_size=0.2):
    g = LP_datos()
    x = np.array([lp.get_psd(fq_band=(0.5,10), nperseg=512, normalize=True)[1] for lp in g])
    x_train, x_test = train_test_split(x, test_size=test_size)
    return x_train, x_test


def data_preprocesados(n=2, n_components=False, include_categorical=False, onehot=False):
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

    if n_components and isinstance(n_components, int):
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())
    
    return X


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


def etiquetas():
    file_name = os.path.join(path_file, 'pares2.csv')

    ldl = []
    with open(file_name, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0][0] != '#':
                li = list(map(int, row))
                ldl.append(li)
    
    LP_index = combine_list(ldl)
    [lpi.sort() for lpi in LP_index]
    label_size = list(map(len, LP_index))
    label = [[n+1]*ls for n, ls in enumerate(label_size)]

    index_list = list(itertools.chain(*LP_index))
    label_list = list(itertools.chain(*label))

    return np.array(index_list), np.array(label_list)


def entrenables(**kwargs):
    lp_idx, lp_labels = etiquetas()
    X = data_preprocesados(**kwargs)

    X_labeled = np.empty((len(lp_idx), X.shape[1]))
    for n, idx in enumerate(lp_idx):
        X_labeled[n,:] = X[idx,:]

    return X_labeled, np.array(lp_labels)