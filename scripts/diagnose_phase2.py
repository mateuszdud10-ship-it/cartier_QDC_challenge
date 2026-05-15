# Run this in your cartier_QDC_challenge root as:
# python scripts/diagnose_phase2.py

import os, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATDIR = os.path.join(ROOT, "data", "features")

PROTECTED = ["CLIENT_ID", "DATE_TARGET", "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
             "LOG_TARGET_3Y", "LOG_TARGET_5Y", "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(FEATDIR, "train_features_final.csv"), low_memory=False)
test  = pd.read_csv(os.path.join(FEATDIR, "test_features_final.csv"),  low_memory=False)
train["DATE_TARGET"] = pd.to_datetime(train["DATE_TARGET"])
test["DATE_TARGET"]  = pd.to_datetime(test["DATE_TARGET"])

# ── Savings rate merge (same as V3) ─────────────────────────────────────────
sr_path = os.path.join(ROOT, "data", "raw", "savings_rate.csv")
sr = pd.read_csv(sr_path, parse_dates=["Date"])
sr["__snap_year__"] = sr["Date"].dt.year
sr_annual = sr.groupby("__snap_year__")["SavingsRate"].mean().reset_index()
sr_annual.columns = ["__snap_year__", "SAVINGS_RATE_AT_SNAPSHOT"]
global_median_sr = float(sr_annual["SAVINGS_RATE_AT_SNAPSHOT"].median())

train.drop(columns=["SAVINGS_RATE_AT_SNAPSHOT"], errors="ignore", inplace=True)
test.drop(columns=["SAVINGS_RATE_AT_SNAPSHOT"],  errors="ignore", inplace=True)
train["__snap_year__"] = train["DATE_TARGET"].dt.year
test["__snap_year__"]  = test["DATE_TARGET"].dt.year
train = train.merge(sr_annual, on="__snap_year__", how="left")
test  = test.merge(sr_annual,  on="__snap_year__", how="left")
train["SAVINGS_RATE_AT_SNAPSHOT"].fillna(global_median_sr, inplace=True)
test["SAVINGS_RATE_AT_SNAPSHOT"].fillna(global_median_sr,  inplace=True)
train.drop(columns=["__snap_year__"], errors="ignore", inplace=True)
test.drop(columns=["__snap_year__"],  errors="ignore", inplace=True)

# ── Target resolution ────────────────────────────────────────────────────────
if "TARGET_5Y" not in train.columns and "TARGET_3Y" in train.columns:
    train["TARGET_5Y"]        = train["TARGET_3Y"]
    train["LOG_TARGET_5Y"]    = train.get("LOG_TARGET_3Y", np.log1p(train["TARGET_3Y"]))
    train["BINARY_TARGET_5Y"] = train.get("BINARY_TARGET_3Y", (train["TARGET_3Y"]>0).astype(int))
    test["TARGET_5Y"]         = test["TARGET_3Y"]
    test["LOG_TARGET_5Y"]     = test.get("LOG_TARGET_3Y",  np.log1p(test["TARGET_3Y"]))
    test["BINARY_TARGET_5Y"]  = test.get("BINARY_TARGET_3Y",  (test["TARGET_3Y"]>0).astype(int))

PROTECTED = [c for c in ["CLIENT_ID","DATE_TARGET","TARGET_3Y","TARGET_5Y","TARGET_10Y",
             "LOG_TARGET_3Y","LOG_TARGET_5Y","BINARY_TARGET_3Y","BINARY_TARGET_5Y"]
             if c in train.columns]
FEAT_COLS = [c for c in train.columns if c not in PROTECTED]

# ── Build matrices ───────────────────────────────────────────────────────────
cat_cols = [c for c in FEAT_COLS if train[c].dtype == object]
num_cols = [c for c in FEAT_COLS if train[c].dtype != object]
print(f"cat_cols: {len(cat_cols)}  |  num_cols: {len(num_cols)}  |  total: {len(FEAT_COLS)}")

enc    = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
tr_cat = enc.fit_transform(train[cat_cols].fillna("missing"))
te_cat = enc.transform(test[cat_cols].fillna("missing"))

X_train_df = pd.concat([train[num_cols].reset_index(drop=True),
                         pd.DataFrame(tr_cat, columns=cat_cols)], axis=1)

# ── Duplicate check ──────────────────────────────────────────────────────────
dupes = X_train_df.columns[X_train_df.columns.duplicated()].tolist()
print(f"\nDuplicate column names: {dupes if dupes else 'NONE'}")

# ── All-NaN column check (THE key diagnostic) ────────────────────────────────
print("\nColumns with ANY NaN:")
nan_counts = X_train_df.isnull().sum()
all_nan_cols = nan_counts[nan_counts == len(X_train_df)].index.tolist()
print(f"  ALL-NaN columns (dropped by SimpleImputer): {all_nan_cols if all_nan_cols else 'NONE'}")

high_nan = nan_counts[nan_counts > len(X_train_df) * 0.99]
if len(high_nan):
    print(f"  Columns with >99% NaN (near-all-NaN):")
    for col, cnt in high_nan.items():
        print(f"    {col}: {cnt:,} NaN ({cnt/len(X_train_df):.2%})")

# ── Imputer behaviour check ──────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_df)
print(f"\nX_train_df shape : {X_train_df.shape}")
print(f"X_train_imp shape: {X_train_imp.shape}")
print(f"Columns lost by imputer: {X_train_df.shape[1] - X_train_imp.shape[1]}")

# ── Which column(s) did the imputer drop? ────────────────────────────────────
# SimpleImputer keeps an internal indicator of which features were kept
# via the statistics_ attribute — NaN statistics_ means the column was all-NaN
if hasattr(imputer, 'statistics_'):
    dropped_idx = [i for i, s in enumerate(imputer.statistics_) if np.isnan(s)]
    if dropped_idx:
        print(f"\nIMPUTER DROPPED these column indices: {dropped_idx}")
        for i in dropped_idx:
            print(f"  idx {i} → column name: '{X_train_df.columns[i]}'")
            print(f"           NaN count: {X_train_df.iloc[:, i].isnull().sum():,}")
            print(f"           Non-NaN count: {X_train_df.iloc[:, i].notna().sum():,}")
    else:
        print("\nNo columns dropped by imputer (statistics_ all finite)")

print("\nDIAGNOSIS COMPLETE")
