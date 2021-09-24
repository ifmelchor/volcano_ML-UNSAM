#!/usr/bin/env python3
# coding=utf-8

from .generaDB import Generar

if __name__ == '__main__':
    g = Generar('../dataset/MicSigV1_v1_1.json')

    for i in range(len(g.LPs)):
        

