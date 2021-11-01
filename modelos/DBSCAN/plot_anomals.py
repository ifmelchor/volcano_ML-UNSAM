#!/usr/bin/env python3
# coding=utf-8

import sys
sys.path.append( '../..' )
from utils import LP_datos

gen = LP_datos()

LP_anomalos = [206, 211, 365, 369, 446, 627, 945]

for LP in LP_anomalos:
    gen[LP].plot()

