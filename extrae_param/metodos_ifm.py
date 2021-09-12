#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

class Extraer(object):
    def __init__(self, data, fs=100):
        self.data = data
        self.fs = fs
        self.duration = len(data)/fs
    

    def get_psd(self, fq_band=(), nperseg=1024, noverlap=0.75, db_scale=False):
        """ 
        Devuelve la PSD de la señal
        """

        freq, PSD = signal.welch(self.data, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        if fq_band:
            f_min = fq_band[0]
            f_max = fq_band[1]
            f_min_pos = np.argmin(np.abs(freq-f_min))
            f_max_pos = np.argmin(np.abs(freq-f_max))
            freq = freq[f_min_pos:f_max_pos]
            PSD = PSD[f_min_pos:f_max_pos]
        
        if db_scale:
            PSD = 10*np.log10(PSD)
        
        return freq, PSD
    

    def get_fq(self, **kwargs):
        """ 
        Devuelve la frecuencia dominante
        """

        f, psd = self.get_psd(**kwargs)

        fq_dominant = f[np.argmax(psd)]

        return fq_dominant
    

    def get_fq_centroid(self, **kwargs):
        """
        Devuelve la frecuencia centroide o baricentro de la PSD
        """
        f, psd = self.get_psd(**kwargs)

        return np.sum(f*psd)/np.sum(psd)
    

    def plot(self, what='psd', **kwargs):
        if what == 'psd':
            fig, ax = plt.subplots(1, 1)
            f, PSD = self.get_psd(**kwargs)
            ax.plot(f, PSD, color='k', lw=1.1)
            ax.grid(True)
            ax.set_xlabel('Freq. [Hz]')
            ax.set_ylabel('PSD')
            plt.show()
            plt.close()
        
        elif what == 'wave':
            fig, ax = plt.subplots(1, 1)
            x = np.linspace(0, self.duration, len(self.data))
            ax.plot(x, self.data, color='k', lw=1.1)
            ax.set_xlabel('Time [sec]')
            ax.set_ylabel('cnts')
            plt.show()
            plt.close()

        else:
            return


    def get_peaks(self, threshold=0.5, **kwargs):
        """
        Devuelve un diccionario con los picos dominantes
        """
        f, psd = self.get_psd(**kwargs)

        psd_max = psd.max()
        psd_threshold = psd_max * threshold

        peaks = signal.find_peaks(psd, height=psd_threshold)
        fq_peaks = f[peaks[0]]
        peak_width = signal.peak_widths(psd, peaks[0])
        
        peak_dict = {}

        for i in range(len(peaks[0])):
            peak_dict[i] = {
                'fq':fq_peaks[i],
                'psd':peaks[1]['peak_heights'][i],
                'width':peak_width[0][i]
            }

        return peak_dict


    def get_pentropy(self, fq_band=(), order=5, tau=1):
        try:
            from antropy import perm_entropy
        except:
            raise ImportError(' AntroPy >>> install from https://raphaelvallat.com/antropy/')
        
        if fq_band:
            return
        else:
            data = self.data
        
        h = perm_entropy(data, order=order, delay=tau, normalize=True)

        return h


# aca las probamos con el primer LP
if __name__ == '__main__':

    dset = '../dataset/MicSigV1_v1_1.json'
    df = pd.read_json(dset)

    types = df['Type']
    LPs = df[types == 'LP']

    LP1 = LPs.iloc[0]
    data = LP1.Data[LP1.StartPoint:LP1.EndPoint]

    ext = Extraer(data)
    ans = ext.get_peaks(fq_band=(1,10))
    ext.plot(fq_band=(1,10))



