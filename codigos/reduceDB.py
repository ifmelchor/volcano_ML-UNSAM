#!/usr/bin/env python3
# coding=utf-8

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import sys

dset = sys.argv[1]
df = pd.read_json(dset)

## PREPROCESING
cat1_attr = df[""]
cat2_attr = df[""]
num_attr = df.drop("")

full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), num_attr),
    ("cat_1", OneHotEncoder(), cat1_attr),
    ("cat_2", OneHotEncoder(), cat2_attr)
])

dset_prepared = full_pipeline.fit_transform(df)

## REDUCING