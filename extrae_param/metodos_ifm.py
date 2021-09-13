#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

try:
    from antropy import perm_entropy
except ImportError:
    print(' instala AntroPy [https://raphaelvallat.com/antropy] pip3 install antropy')

try:
    import pywt
except ImportError:
    print(' instala PyWavelet [https://github.com/PyWavelets/pywt] pip3 install PyWavelets')


class Extraer(object):
    def __init__(self, data, fs=100):
        self.data = signal.detrend(data)
        self.fs = fs
        self.duration = len(data)/fs
        self.time = np.linspace(0, self.duration, len(self.data))
    

    def filter_data(self, fq_band=(), order=5):
        if fq_band:
            nyq = (self.fs/2)
            low = fq_band[0] / nyq
            high = fq_band[1] / nyq
            b, a = signal.butter(order, [low, high], btype='band')
            data = signal.lfilter(b, a, self.data)
            return data
        else:
            return self.data


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
            ax.plot(self.time, self.data, color='k', lw=1.1)
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


    def get_pentropy(self, order=5, tau=1, **filter_kwargs):
        """
        Devuelve la entropia de permutación de la señal.
        """
        
        # filtra la señal
        if filter_kwargs.get('fq_band', None):
            data = self.filter_data(**filter_kwargs)
        else:
            data = self.data
        
        h = perm_entropy(data, order=order, delay=tau, normalize=True)

        return h
    

    def wavelet_test(self, **kwargs):
        out = []
        l_min = kwargs.get('level_min', 4)
        l_max = kwargs.get('level_max', 8)

        for l in np.arange(l_min, l_max+1):
            for wlet in pywt.wavelist('db'):
                max_val, freq = self.wavelet_decompose(wlet, l, **kwargs)
                out += [(wlet, l, max_val, freq)]
        return out
    

    def best_wavelet_fit(self, n=1, **kwargs):
        ans = self.wavelet_test(**kwargs)
        ans.sort(key=lambda x: x[2], reverse=True)
        return ans[0:n]


    def wavelet_decompose(self, wavelet_name, level, mode='mean'):
        wavelet = pywt.Wavelet(wavelet_name)
        wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
        nodes = wp.get_level(level, order='freq')
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)
        freq = np.linspace(0, self.fs/2, len(nodes))

        if mode == 'mean':
            PSD = values.mean(axis=1)
        
        elif mode == 'min':
            PSD = values.min(axis=1)
        
        elif mode == 'max':
            PSD = values.max(axis=1)
        
        else:
            raise ValueError(' mode must be "mean", "min" or "max"')
            
        return PSD.max(), freq[np.argmax(PSD)]


# aca las probamos con el primer LP
if __name__ == '__main__':

    dset = '../dataset/MicSigV1_v1_1.json'
    df = pd.read_json(dset)

    types = df['Type']
    LPs = df[types == 'LP']

    LP1 = LPs.iloc[0]
    data = LP1.Data[LP1.StartPoint:LP1.EndPoint]

    ext = Extraer(data)

    # parametros relacionados con la PSD
    att1 = ext.get_peaks(threshold=0.5) # aca se puede jugar con el threshold

    att2 = ext.get_fq_centroid()
    
    # parametros relacionados con la forma de onda
    att3 = ext.get_pentropy() # aca se puede jugar con los ordenes y los tau

    # parametros relacionados con la transformada wavelet
    att4 = ext.best_wavelet_fit(n=5, mode='max') # aca se puede jugar con el modo y el n


    # inspeccioná la clase y los att


