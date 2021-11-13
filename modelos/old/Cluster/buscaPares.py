#!/usr/bin/env python3
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append( '../../..' )
from generaDB import  Generar

def plot_model(dict_in):
    row =int(np.ceil(len(dict_in)/3))

    fig, axes = plt.subplots(row, 3, figsize=(16,9))
    gen = Generar()

    for n, ax in enumerate(axes.reshape(axes.size,)):
        lp_list = dict_in.get(n+1)
        if lp_list:
            ax.set_title(n+1, size=12)
            for lp in lp_list:
                freq, PSD = gen[lp].get_psd((1,6), normalize=True)
                ax.plot(freq, PSD, label=lp)
            ax.legend()
        else:
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])

    return fig

if __name__ == '__main__':
    df = pd.read_json('./kmeans_labels.json')
    labels = df['Label']
    
    while True:
        eventos = {}
        for i in range(1, 6):
            clase = df[labels == i]
            eventos[i]=list(clase.Index.iloc[np.random.randint(0, high=len(clase), size=3)])
            
        fig = plot_model(eventos)
        fig.suptitle('Modelo KMeans 5 clusters')
        plt.show()

        plt.close()

