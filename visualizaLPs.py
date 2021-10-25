#!/usr/bin/env python3
# coding=utf-8

from generaDB import Generar
import matplotlib.pyplot as plt

gen = Generar()

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

def showLPs(lp_list):
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
    return fig

for p in par:
    fig = showLPs(map(gen.get, p))
    file_name = '_'.join(map(str, p))
    fig.savefig('./imag/LP_%s.png' % file_name)
    plt.close()
    
