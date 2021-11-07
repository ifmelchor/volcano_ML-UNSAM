#!/usr/bin/env python3
# coding=utf-8

from generaDB import Generar
import matplotlib.pyplot as plt
import pandas as pd

gen = Generar()
json_file = './dataset/GMM_4cluster.json'
df = pd.read_json(json_file)


