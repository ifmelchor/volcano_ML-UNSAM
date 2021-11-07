#!/usr/bin/env python3
# coding=utf-8

from generaDB import Generar
from utils import plot_LP_list
import matplotlib.pyplot as plt

gen = Generar()

# Lista de posibles pares mediante la inspeccion visual del procesado del KMeans
par = [
    (168,688),
    (685,698),
    (12,300),
    (684,1032),
    (300,777),
    (324,142),
    (112,869),
    (100,394),
    (178,684),
    (147,567,392),
    (226,649),
    (253,476),
    (264,30),
    (98,340),
    (237,772),
    (420,6),
    (248,72)
]

for p in par:
    fig = plot_LP_list(map(gen.get, p))
    file_name = '_'.join(map(str, p))
    fig.savefig('./imag/LP_%s.png' % file_name)
    plt.close()
