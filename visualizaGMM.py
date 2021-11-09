#!/usr/bin/env python3
# coding=utf-8

from generaDB import Generar
from utils import plot_LP_list

import pandas as pd
import numpy as np
import sys

gen = Generar()
json_file = './dataset/GMM_4cluster.json'
df = pd.read_json(json_file)

k = int(sys.argv[1])
df_k = df[df['Label']==k]
size = df_k.Index.shape[0]
print(f' Label {k}: {size} LPs')

nro_events = 3

while True:
    index = df_k.Index.to_numpy()[np.random.choice(size, 2)]
    LP_list = map(gen.get, index)
    fig = plot_LP_list(LP_list)


