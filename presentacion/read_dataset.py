#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Archivo de lectura de la base de datos

@author: Ivan Melchor (ifmelchor@unrn.edu.ar)
"""

import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

json_file = '../dataset/MicSigV1_v1_1.json'
df = pd.read_json(json_file)

# LP list
types = df['Type']
LPs = df[types == 'LP']
nro_LP = len(LPs)

print(LPs.iloc[643])
print(LPs.iloc[657])

# years
# print(df['Year'].unique())
# duraciones = []
# fq_dominantes = []
# energia_fq_dominante = []
# energia_total = []

# for i in range(len(LPs)):
#     LPi = LPs.iloc[i]
#     duraciones.append(LPi.Duration)
#     LP_waveform = LPi.Data[LPi.StartPoint:LPi.EndPoint]
#     f1, PSD1 = signal.welch(LP_waveform, LPi.SampleRate, nperseg=1024, scaling='density')
#     max_fq = f1[np.argmax(PSD1)]
#     energia_fq_dominante.append(np.max(PSD1))
#     fq_dominantes.append(max_fq)
#     energia_total.append(sum(PSD1))

# plt.hist(fq_dominantes, 'scott')
# plt.ylabel('Frecuencia')
# plt.xlabel('Frecuencia dominante [Hz]')
# plt.show()

# plt.hist(duraciones, 'scott')
# plt.ylabel('Frecuencia')
# plt.xlabel('Duración del evento [sec]')
# plt.show()

# plt.hist(10*np.log10(energia_fq_dominante), 'scott')
# plt.ylabel('Frecuencia')
# plt.xlabel(r'Energía dominante [cnts$^2$/Hz dB]')
# plt.show()

# plt.hist(10*np.log10(energia_total), 'scott')
# plt.ylabel('Frecuencia')
# plt.xlabel(r'Energía total [cnts$^2$ dB]')
# plt.show()

# while True:
#     fig, axes = plt.subplots(10, 2, figsize=(19, 20), gridspec_kw=dict(width_ratios=[10,3], hspace=0.5))
#     for k in range(10):
#         i = np.random.randint(0, 1044)
#         LP_k = LPs.iloc[i]
#         LP_waveform = LP_k.Data[LP_k.StartPoint:LP_k.EndPoint]
#         N_points = len(LP_waveform)
#         time = np.linspace(0, LP_k.Duration, N_points)
#         axes[k][0].plot(time, LP_waveform, 'k', lw=0.9)
#         axes[k][0].set_ylabel(r'LP$_{%s}$' %i)
#         axes[k][0].set_xlabel('segundos')
#         axes[k][0].set_xlim(0, LP_k.Duration)

#         nps = 1024
#         if N_points < nps:
#             nps = 512

#         f, PSD = signal.welch(LP_waveform, LP_k.SampleRate, nperseg=nps, scaling='density')
#         axes[k][1].plot(f, PSD, color='k')
#         axes[k][1].set_xlabel('freq. [Hz]')
#         axes[k][1].set_ylabel(r'PSD')
#         axes[k][1].grid()
#         axes[k][1].set_xlim(0, 10)
#     plt.show()
#     plt.close()