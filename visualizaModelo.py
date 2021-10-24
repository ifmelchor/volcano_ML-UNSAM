#!/usr/bin/env python3
# coding=utf-8

from generaDB import Generar
import pandas as pd

if __name__ == '__main__':
    gen = Generar()
    etiquetas = pd.read_json('./modelos/KMeans/kmeans_labels.json')
    
    # plotear primer LP
    gen[0].plot()


