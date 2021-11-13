#!/usr/bin/env python3
# coding=utf-8

import csv
import random
import numpy as np
from itertools import chain
from generaDB import Generar
from utils import combine_list, plot_LP_list

ldl = []
with open('pares2.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0][0] != '#':
            li = list(map(int, row))
            ldl.append(li)

labels = combine_list(ldl)
print('Number of lists/labels: ', len(labels))

# take one
def take_one(size):
    return random.randint(0,size-1)

label_list = list(chain(*labels))
label_list_size = list(map(len, labels))

# comapare between labels
labels_index_rand = list(map(take_one, label_list_size))
LPindex_test = [lab[i] for lab,i in zip(labels, labels_index_rand)]
LPindex_test.sort()
print(LPindex_test)
LPindex_test = np.array(LPindex_test)

# # #compare intra labels
# LPindex_test = np.array(labels[1])

# gen = Generar()
# while True:
#     index = LPindex_test[np.random.choice(LPindex_test.shape[0], 3)]
#     LP_list = map(gen.get, index)
#     fig = plot_LP_list(LP_list)