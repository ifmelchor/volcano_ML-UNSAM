#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Archivo de lectura de la base de datos

@author: Ivan Melchor (ifmelchor@unrn.edu.ar)
"""

import pandas as pd

json_file = './dataset/MicSigV1_v1_1.json'
df = pd.read_json(json_file)

