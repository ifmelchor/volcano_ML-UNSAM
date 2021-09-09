#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import pandas as pd

# Aca definimos las diferentes funciones de extracción de parametros
def func1(data):
    return np.mean(data)




# aca las probamos con el primer LP
if __name__ == '__main__':

    dset = '../dataset/MicSigV1_v1_1.json'
    df = pd.read_json(dset)

    types = df['Type']
    LPs = df[types == 'LP']

    LP1 = LPs.iloc[0]
    data = LP1.Data[LP1.StartPoint:LP1.EndPoint]

    param1 = func1(data)



