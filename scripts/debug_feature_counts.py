import os, sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")

train = pd.read_csv(os.path.join(FEAT_DIR, "train_features_final.csv"), low_memory=False)
print('train shape', train.shape)

ID_COLS = ["CLIENT_ID", "DATE_TARGET"]
TARGET_COLS = ["TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
               "LOG_TARGET_3Y", "LOG_TARGET_5Y",
               "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]

feat_cols = [c for c in train.columns if c not in ID_COLS + TARGET_COLS]
print('len(feat_cols)=', len(feat_cols))
print('feat_cols sample 10 =', feat_cols[:10])
print('TO_BTQ in feat_cols =', 'TO_BTQ' in feat_cols)
print('feat_cols last 10 =', feat_cols[-10:])
print('unique len feat_cols =', len(set(feat_cols)))
print('duplicate names =', [c for c in feat_cols if feat_cols.count(c) > 1])
print('columns types:')
print(train[feat_cols].dtypes.value_counts())
print('shape selected:', train[feat_cols].shape)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(train[feat_cols])
print('shape after impute:', X.shape)

lr_feat_cols = [c for c in feat_cols if c != 'TO_BTQ']
print('len(lr_feat_cols)=', len(lr_feat_cols))
Xlr = imp.fit_transform(train[lr_feat_cols])
print('shape Xlr after impute:', Xlr.shape)
print('sample dtypes for lr columns:')
print(train[lr_feat_cols].dtypes[train[lr_feat_cols].dtypes==object].index.tolist())
