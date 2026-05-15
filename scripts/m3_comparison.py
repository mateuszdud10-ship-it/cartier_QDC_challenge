"""
=============================================================================
M3 COMPARISON SCRIPT — m3_comparison.py
Cartier QTEM Data Challenge
=============================================================================

PURPOSE
-------
This script is a DECISION TOOL, not a final model. It systematically compares
improvement options for the M3 CLV model on a manageable subsample (the 2018
training snapshot, ~350K rows) before committing to any change in the full
M3 script. Nothing is saved permanently — all comparisons are printed to
stdout and written to output/tables/m3_comparison_results.csv.

WHAT IS COMPARED
----------------
Dimension 1 — Regressor Architecture (most impactful):
  A) Two-Stage current  : XGBoost regressor on log-target, positives only
                          back-transform via expm1() [CURRENT BASELINE]
  B) Tweedie            : Single XGBoost reg:tweedie on ALL clients (zeros
                          included), handles zero-inflation natively
  C) Soft-weighted      : Two-stage but classifier P(active) used as
                          sample_weight on the full training set (softens
                          the hard positive/negative boundary)

Dimension 2 — Smearing correction:
  A) None (current)     : exp(pred) directly [systematic underestimate]
  B) Log-normal smear   : exp(pred + sigma^2/2) where sigma^2 is the
                          residual variance on val positives

Dimension 3 — Classifier hyperparameter tuning:
  A) Current config     : max_depth=6, min_child_weight=10, subsample=0.8
  B) Tuned config       : grid search over depth/min_child/subsample
                          (4 combinations, val PR-AUC as criterion)

Dimension 4 — Feature set for regressor:
  A) Full 111 features  : all V2 features (current approach)
  B) Reduced set        : drop ALL_* sequence features from regressor only
                          (they help the classifier but may add noise to
                          the regressor — tested empirically here)

EVALUATION METRICS
------------------
  Classifier  : PR-AUC, Recall top-10% on val (2015 snapshot)
  Regressor   : Spearman rank r, RMSE log-space, Revenue capture top 1%/5%/10%
                on val (2015 snapshot positives)

SUBSAMPLE RATIONALE
-------------------
We use only the 2018 training snapshot to keep runtime under ~20 minutes
while maintaining representative data size (~350K rows, same feature
distribution as the full train). The val fold is always the 2015 snapshot
(same as in M3), the test is always the 2021 snapshot.

HOW TO USE THE RESULTS
-----------------------
After running this script, read the RECOMMENDED CONFIG FOR M3 V2 section
printed at the end and carry the winning configuration into m3_clv_v2.py.

Run with:
    python scripts/m3_comparison.py
=============================================================================
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              mean_squared_error, mean_absolute_error)
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT, "data", "features")
OUTPUT_DIR   = os.path.join(ROOT, "output", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROTECTED = ["CLIENT_ID", "DATE_TARGET",
             "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
             "LOG_TARGET_3Y", "LOG_TARGET_5Y",
             "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]

# ---------------------------------------------------------------------------
# Helper: booster feature importance (XGBoost 3.x compatible)
# ---------------------------------------------------------------------------
def booster_importance(model, feat_names, importance_type="gain"):
    scores = model.get_booster().get_score(importance_type=importance_type)
    rows = []
    for k, v in scores.items():
        try:
            idx   = int(k.lstrip("f"))
            fname = feat_names[idx] if idx < len(feat_names) else k
        except (ValueError, IndexError):
            fname = k
        rows.append({"feature": fname, "importance": v})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        {"feature": feat_names, "importance": [0.0] * len(feat_names)})
    return df.sort_values("importance", ascending=False)

# ---------------------------------------------------------------------------
# Helper: revenue capture at multiple percentiles
# ---------------------------------------------------------------------------
def revenue_capture(y_true, scores, pcts=(0.01, 0.05, 0.10)):
    total = y_true.sum()
    if total == 0:
        return {p: 0.0 for p in pcts}
    order = np.argsort(scores)[::-1]
    res   = {}
    for p in pcts:
        n_top    = max(1, int(len(y_true) * p))
        res[p]   = float(y_true[order[:n_top]].sum() / total)
    return res

# ---------------------------------------------------------------------------
# Phase 0 — Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("  M3 COMPARISON SCRIPT — Cartier QTEM Data Challenge")
print("=" * 70)
print(f"XGBoost version: {xgb.__version__}")

train_path = os.path.join(FEATURES_DIR, "train_features_final.csv")
test_path  = os.path.join(FEATURES_DIR, "test_features_final.csv")
if not os.path.exists(train_path):
    print(f"ERROR: {train_path} not found. Run fe_v2_pipeline.py first.")
    sys.exit(1)

print("\nLoading train and test sets...")
train = pd.read_csv(train_path, low_memory=False)
test  = pd.read_csv(test_path,  low_memory=False)
train["DATE_TARGET"] = pd.to_datetime(train["DATE_TARGET"])
test["DATE_TARGET"]  = pd.to_datetime(test["DATE_TARGET"])
print(f"  Full train: {train.shape}")
print(f"  Test:       {test.shape}")

# ---------------------------------------------------------------------------
# Phase 1 — Subsample: keep only 2018 snapshot from train
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 1 — SUBSAMPLE (2018 snapshot only)")
print("=" * 70)

sub = train[train["DATE_TARGET"].dt.year == 2018].copy()
print(f"  Subsample size (2018 snapshot): {len(sub):,} rows")
print(f"  (Full train has {len(train):,} rows — using {len(sub)/len(train):.1%})")

if len(sub) == 0:
    print("  WARNING: no 2018 rows found. Falling back to full train.")
    sub = train.copy()

# Validation split: 2015 snapshot from FULL train (not sub)
# so val is always the same fold as in the final M3
val_full  = train[train["DATE_TARGET"].dt.year == 2015].copy()
print(f"  Val fold (2015 snapshot, from full train): {len(val_full):,} rows")

# Combined sub + val for feature preprocessing
combined  = pd.concat([sub, val_full], ignore_index=True)
print(f"  Combined sub+val for preprocessing: {len(combined):,} rows")

FEAT_COLS    = [c for c in train.columns if c not in PROTECTED]
cat_cols     = [c for c in FEAT_COLS if train[c].dtype == object]
num_cols     = [c for c in FEAT_COLS if train[c].dtype != object]
print(f"\n  Total features: {len(FEAT_COLS)} ({len(cat_cols)} categorical, {len(num_cols)} numeric)")

# Feature set B for regressor (drop ALL_* sequence features)
ALL_PREFIXES = ("HAS_ALL_", "N_TOTAL_ALL_", "N_DISTINCT_ALL_", "DIVERSITY_ALL_")
reduced_feat  = [c for c in FEAT_COLS
                 if not any(c.startswith(p) for p in ALL_PREFIXES)]
print(f"  Reduced feature set (no ALL_* for regressor): {len(reduced_feat)} features")
print(f"  Dropped ALL_* features: {len(FEAT_COLS) - len(reduced_feat)}")

# ---------------------------------------------------------------------------
# Phase 2 — Preprocessing (fit on sub only, transform all)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 2 — PREPROCESSING")
print("=" * 70)

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(sub[cat_cols].fillna("__missing__"))

def make_X(df, feat_list, enc_obj, imp_obj=None, fit_imp=False):
    """Encode, impute and return numpy array for a given feature list."""
    _cat  = [c for c in cat_cols if c in feat_list]
    _num  = [c for c in num_cols if c in feat_list]
    cat_m = pd.DataFrame(enc_obj.transform(df[_cat].fillna("__missing__")),
                         columns=_cat) if _cat else pd.DataFrame()
    num_m = df[_num].reset_index(drop=True) if _num else pd.DataFrame()
    X_df  = pd.concat([num_m, cat_m], axis=1)
    if fit_imp:
        imp_obj.fit(X_df)
    return imp_obj.transform(X_df), list(X_df.columns)

# Build imputers (one per feature set, fit on sub train positives / full sub)
imp_full    = SimpleImputer(strategy="median")
imp_reduced = SimpleImputer(strategy="median")

# Masks
sub_val_dates  = combined["DATE_TARGET"]
val_mask_comb  = (sub_val_dates.dt.year == 2015).values
trn_mask_comb  = ~val_mask_comb

y_comb_bin = combined["BINARY_TARGET_5Y"].values
y_comb_log = combined["LOG_TARGET_5Y"].values
y_comb_raw = combined["TARGET_5Y"].values

# Full feature set
X_full, feat_names_full = make_X(combined, FEAT_COLS, enc, imp_full, fit_imp=True)
# Reduced feature set
X_red,  feat_names_red  = make_X(combined, reduced_feat, enc, imp_reduced, fit_imp=True)

# Test set transforms
X_test_full, _ = make_X(test, FEAT_COLS,    enc, imp_full)
X_test_red,  _ = make_X(test, reduced_feat, enc, imp_reduced)
y_test_bin     = test["BINARY_TARGET_5Y"].values
y_test_log     = test["LOG_TARGET_5Y"].values
y_test_raw     = test["TARGET_5Y"].values

print(f"  X_full shape:    {X_full.shape}")
print(f"  X_reduced shape: {X_red.shape}")
print(f"  X_test_full:     {X_test_full.shape}")
print(f"  Train (sub) mask: {trn_mask_comb.sum():,} | Val mask: {val_mask_comb.sum():,}")

# Short aliases
trn  = trn_mask_comb
val  = val_mask_comb

y_trn_bin = y_comb_bin[trn]
y_val_bin = y_comb_bin[val]
y_trn_log = y_comb_log[trn]
y_val_log = y_comb_log[val]
y_trn_raw = y_comb_raw[trn]
y_val_raw = y_comb_raw[val]

# Positive masks
pos_trn = y_trn_bin == 1
pos_val = y_val_bin == 1

print(f"\n  Train positives: {pos_trn.sum():,} ({pos_trn.mean():.1%})")
print(f"  Val positives:   {pos_val.sum():,} ({pos_val.mean():.1%})")
print(f"  Test positives:  {y_test_bin.sum():,} ({y_test_bin.mean():.1%})")
scale_pos = (1 - y_trn_bin.mean()) / y_trn_bin.mean()
print(f"  scale_pos_weight (train sub): {scale_pos:.2f}")

# ---------------------------------------------------------------------------
# Phase 3 — Dimension 3: Classifier hyperparameter comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 3 — DIM 3: CLASSIFIER HYPERPARAMETER TUNING")
print("=" * 70)

clf_configs = {
    "Current": dict(max_depth=6, min_child_weight=10, subsample=0.8),
    "Tuned-A": dict(max_depth=4, min_child_weight=5,  subsample=0.7),
    "Tuned-B": dict(max_depth=4, min_child_weight=10, subsample=0.9),
    "Tuned-C": dict(max_depth=6, min_child_weight=5,  subsample=0.9),
}

clf_results   = {}
best_clf_name = "Current"
best_clf_prauc = -1
best_clf_model = None

for name, cfg in clf_configs.items():
    print(f"\n  Training classifier [{name}]: {cfg}")
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        **cfg,
    )
    clf.fit(
        X_full[trn], y_trn_bin,
        eval_set=[(X_full[val], y_val_bin)],
        verbose=False,
    )
    y_prob_val  = clf.predict_proba(X_full[val])[:, 1]
    prauc_val   = average_precision_score(y_val_bin, y_prob_val)
    roc_val     = roc_auc_score(y_val_bin, y_prob_val)
    n_val       = len(y_val_bin)
    top10_idx   = np.argsort(y_prob_val)[::-1][:n_val // 10]
    rec10_val   = y_val_bin[top10_idx].sum() / max(1, y_val_bin.sum())

    clf_results[name] = {
        "PR-AUC val":       round(prauc_val, 4),
        "ROC-AUC val":      round(roc_val, 4),
        "Recall@10% val":   round(rec10_val, 4),
        "Trees used":       clf.best_iteration + 1,
        **cfg,
    }
    print(f"    PR-AUC val={prauc_val:.4f}  ROC-AUC={roc_val:.4f}"
          f"  Recall@10%={rec10_val:.4f}  trees={clf.best_iteration+1}")

    if prauc_val > best_clf_prauc:
        best_clf_prauc = prauc_val
        best_clf_name  = name
        best_clf_model = clf
        best_clf_cfg   = cfg

print(f"\n  >>> CLASSIFIER WINNER: [{best_clf_name}]"
      f"  PR-AUC val={best_clf_prauc:.4f}")

# ---------------------------------------------------------------------------
# Phase 4 — Dimension 1: Regressor Architecture
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 4 — DIM 1: REGRESSOR ARCHITECTURE")
print("=" * 70)

# Shared classifier probabilities (from best classifier)
p_trn  = best_clf_model.predict_proba(X_full[trn])[:, 1]
p_val  = best_clf_model.predict_proba(X_full[val])[:, 1]
p_test = best_clf_model.predict_proba(X_test_full)[:, 1]

reg_results = {}

# ---- Option A: Two-Stage (current baseline) --------------------------------
print("\n  [A] Two-Stage (positives only, log-target, current baseline)")
reg_A = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)
reg_A.fit(
    X_full[trn][pos_trn], y_trn_log[pos_trn],
    eval_set=[(X_full[val][pos_val], y_val_log[pos_val])],
    verbose=False,
)
pred_log_A_val  = reg_A.predict(X_full[val][pos_val])
pred_eur_A_val  = np.expm1(pred_log_A_val)
rmse_A          = float(np.sqrt(mean_squared_error(y_val_log[pos_val], pred_log_A_val)))
sp_A, _         = spearmanr(y_val_raw[pos_val], pred_eur_A_val)
rc_A            = revenue_capture(y_val_raw[pos_val], pred_eur_A_val)

# Full val CLV score for revenue capture at all clients level
pred_log_A_full = reg_A.predict(X_full[val])
pred_eur_A_full = np.expm1(pred_log_A_full)
clv_A_val       = p_val * pred_eur_A_full
rc_A_all        = revenue_capture(y_val_raw, clv_A_val)

print(f"    Trees: {reg_A.best_iteration+1} | RMSE log={rmse_A:.4f}"
      f" | Spearman(pos)={sp_A:.4f}"
      f" | RevCap@10%(all)={rc_A_all[0.10]:.1%}")

reg_results["A-TwoStage"] = {
    "RMSE log val":          round(rmse_A, 4),
    "Spearman pos val":      round(sp_A, 4),
    "RevCap@1% val(all)":    round(rc_A_all[0.01], 4),
    "RevCap@5% val(all)":    round(rc_A_all[0.05], 4),
    "RevCap@10% val(all)":   round(rc_A_all[0.10], 4),
    "Trees":                 reg_A.best_iteration + 1,
    "Description":           "Two-Stage positives-only (CURRENT)",
}

# ---- Option B: Tweedie on ALL clients --------------------------------------
print("\n  [B] Tweedie (all clients, zero-inflation handled natively)")
reg_B = xgb.XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    eval_metric="tweedie-nloglik@1.5",
    random_state=42,
    n_jobs=-1,
)
reg_B.fit(
    X_full[trn], y_trn_raw,
    eval_set=[(X_full[val], y_val_raw)],
    verbose=False,
)
pred_B_val  = reg_B.predict(X_full[val])
clv_B_val   = p_val * pred_B_val
sp_B_pos, _ = spearmanr(y_val_raw[pos_val], pred_B_val[pos_val])
rc_B_all    = revenue_capture(y_val_raw, clv_B_val)
rmse_B_log  = float(np.sqrt(mean_squared_error(
    np.log1p(y_val_raw[pos_val]), np.log1p(np.maximum(pred_B_val[pos_val], 0)))))

print(f"    Trees: {reg_B.best_iteration+1} | RMSE log(pos)={rmse_B_log:.4f}"
      f" | Spearman(pos)={sp_B_pos:.4f}"
      f" | RevCap@10%(all)={rc_B_all[0.10]:.1%}")

reg_results["B-Tweedie"] = {
    "RMSE log val":          round(rmse_B_log, 4),
    "Spearman pos val":      round(sp_B_pos, 4),
    "RevCap@1% val(all)":    round(rc_B_all[0.01], 4),
    "RevCap@5% val(all)":    round(rc_B_all[0.05], 4),
    "RevCap@10% val(all)":   round(rc_B_all[0.10], 4),
    "Trees":                 reg_B.best_iteration + 1,
    "Description":           "Tweedie on all clients",
}

# ---- Option C: Soft-weighted (full train, P(active) as sample_weight) ------
print("\n  [C] Soft-weighted (all clients, P(active) as sample_weight)")
reg_C = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)
reg_C.fit(
    X_full[trn], y_trn_log,
    sample_weight=p_trn,
    eval_set=[(X_full[val], y_val_log)],
    verbose=False,
)
pred_log_C_val = reg_C.predict(X_full[val])
pred_eur_C_val = np.expm1(pred_log_C_val)
clv_C_val      = p_val * pred_eur_C_val
sp_C_pos, _    = spearmanr(y_val_raw[pos_val], pred_eur_C_val[pos_val])
rmse_C         = float(np.sqrt(mean_squared_error(y_val_log[pos_val],
                                                   pred_log_C_val[pos_val])))
rc_C_all       = revenue_capture(y_val_raw, clv_C_val)

print(f"    Trees: {reg_C.best_iteration+1} | RMSE log(pos)={rmse_C:.4f}"
      f" | Spearman(pos)={sp_C_pos:.4f}"
      f" | RevCap@10%(all)={rc_C_all[0.10]:.1%}")

reg_results["C-SoftWeighted"] = {
    "RMSE log val":          round(rmse_C, 4),
    "Spearman pos val":      round(sp_C_pos, 4),
    "RevCap@1% val(all)":    round(rc_C_all[0.01], 4),
    "RevCap@5% val(all)":    round(rc_C_all[0.05], 4),
    "RevCap@10% val(all)":   round(rc_C_all[0.10], 4),
    "Trees":                 reg_C.best_iteration + 1,
    "Description":           "Soft-weighted (log-target, P(active) weights)",
}

# Winner by Spearman on positives
winner_reg = max(reg_results, key=lambda k: reg_results[k]["Spearman pos val"])
print(f"\n  >>> REGRESSOR ARCHITECTURE WINNER: [{winner_reg}]"
      f"  Spearman pos={reg_results[winner_reg]['Spearman pos val']:.4f}")

# ---------------------------------------------------------------------------
# Phase 5 — Dimension 2: Smearing correction
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 5 — DIM 2: SMEARING CORRECTION (applied to best reg architecture)")
print("=" * 70)

# Use the winning regressor (or A if tie) to test smearing
reg_for_smear_map = {
    "A-TwoStage":    (reg_A, "two-stage"),
    "B-Tweedie":     (reg_B, "tweedie"),
    "C-SoftWeighted":(reg_C, "soft-weighted"),
}
reg_smear, _ = reg_for_smear_map[winner_reg]

# Two-stage scenario (Option A or C): smearing in log-space
if winner_reg in ("A-TwoStage", "C-SoftWeighted"):
    pred_log_val_pos = (reg_A if winner_reg == "A-TwoStage" else reg_C).predict(
        X_full[val][pos_val])
    # sigma^2 = residual variance on val positives
    sigma2 = float(np.var(y_val_log[pos_val] - pred_log_val_pos, ddof=1))
    print(f"  sigma^2 on val positives: {sigma2:.6f}")

    # Without smearing
    pred_eur_no_smear  = np.expm1(pred_log_val_pos)
    mae_no             = float(mean_absolute_error(y_val_raw[pos_val], pred_eur_no_smear))
    med_no             = float(np.median(np.abs(y_val_raw[pos_val] - pred_eur_no_smear)))

    # With log-normal smearing: exp(pred + sigma^2/2)
    pred_eur_smear     = np.exp(pred_log_val_pos + sigma2 / 2)
    mae_sm             = float(mean_absolute_error(y_val_raw[pos_val], pred_eur_smear))
    med_sm             = float(np.median(np.abs(y_val_raw[pos_val] - pred_eur_smear)))

    # Revenue capture in full val (positives pov)
    rc_no = revenue_capture(y_val_raw[pos_val], pred_eur_no_smear)
    rc_sm = revenue_capture(y_val_raw[pos_val], pred_eur_smear)

else:
    # Tweedie: already in EUR-space, smearing not applicable (no log transform)
    pred_pos = reg_B.predict(X_full[val][pos_val])
    mae_no   = float(mean_absolute_error(y_val_raw[pos_val], pred_pos))
    med_no   = float(np.median(np.abs(y_val_raw[pos_val] - pred_pos)))
    mae_sm   = mae_no
    med_sm   = med_no
    rc_no    = revenue_capture(y_val_raw[pos_val], pred_pos)
    rc_sm    = rc_no
    sigma2   = 0.0
    print("  Tweedie winner: no log-space, smearing not applicable — reporting same values.")

smear_results = {
    "No smearing (current)": {
        "MAE EUR val":        round(mae_no, 0),
        "Median AE EUR val":  round(med_no, 0),
        "RevCap@1% pos val":  round(rc_no[0.01], 4),
        "RevCap@5% pos val":  round(rc_no[0.05], 4),
        "RevCap@10% pos val": round(rc_no[0.10], 4),
        "sigma2":             round(sigma2, 6),
    },
    "Log-normal smearing": {
        "MAE EUR val":        round(mae_sm, 0),
        "Median AE EUR val":  round(med_sm, 0),
        "RevCap@1% pos val":  round(rc_sm[0.01], 4),
        "RevCap@5% pos val":  round(rc_sm[0.05], 4),
        "RevCap@10% pos val": round(rc_sm[0.10], 4),
        "sigma2":             round(sigma2, 6),
    },
}
print(f"  No smear : MAE={mae_no:,.0f} EUR  Median AE={med_no:,.0f} EUR"
      f"  RevCap@1%={rc_no[0.01]:.1%}")
print(f"  Smearing : MAE={mae_sm:,.0f} EUR  Median AE={med_sm:,.0f} EUR"
      f"  RevCap@1%={rc_sm[0.01]:.1%}")

smear_better = mae_sm < mae_no or rc_sm[0.01] > rc_no[0.01]
smear_winner = "Log-normal smearing" if smear_better else "No smearing"
print(f"\n  >>> SMEARING WINNER: [{smear_winner}]")

# ---------------------------------------------------------------------------
# Phase 6 — Dimension 4: Feature set for regressor
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 6 — DIM 4: FEATURE SET FOR REGRESSOR")
print("=" * 70)

# Use two-stage architecture (Option A) to isolate the feature set effect
print("  Testing feature set on Two-Stage regressor (Option A architecture)")

print("  [Full 111 features] — already computed above")
sp_full  = reg_results["A-TwoStage"]["Spearman pos val"]
rmse_full = reg_results["A-TwoStage"]["RMSE log val"]
print(f"    Spearman(pos val)={sp_full:.4f} | RMSE log={rmse_full:.4f}")

print("\n  [Reduced feature set — no ALL_*]")
imp_r2 = SimpleImputer(strategy="median")
X_red_trn, _  = make_X(combined[trn], reduced_feat, enc, imp_r2, fit_imp=True)
X_red_val, _  = make_X(combined[val], reduced_feat, enc, imp_r2)

reg_A_red = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)
reg_A_red.fit(
    X_red_trn[pos_trn], y_trn_log[pos_trn],
    eval_set=[(X_red_val[pos_val], y_val_log[pos_val])],
    verbose=False,
)
pred_red_val = reg_A_red.predict(X_red_val[pos_val])
sp_red, _    = spearmanr(y_val_raw[pos_val], np.expm1(pred_red_val))
rmse_red     = float(np.sqrt(mean_squared_error(y_val_log[pos_val], pred_red_val)))
print(f"    Spearman(pos val)={sp_red:.4f} | RMSE log={rmse_red:.4f}")

feat_winner = "Full 111" if sp_full >= sp_red else "Reduced (no ALL_*)"
feat_delta  = sp_full - sp_red
print(f"\n  >>> FEATURE SET WINNER: [{feat_winner}]"
      f"  Delta Spearman={feat_delta:+.4f} (Full - Reduced)")

feat_results = {
    "Full features (111)":     {"Spearman pos val": round(sp_full, 4),
                                 "RMSE log val": round(rmse_full, 4)},
    "Reduced (no ALL_* ~67)":  {"Spearman pos val": round(sp_red, 4),
                                 "RMSE log val": round(rmse_red, 4)},
}

# ---------------------------------------------------------------------------
# Phase 7 — Final comparison table & recommendation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 7 — FINAL COMPARISON TABLE & RECOMMENDATION")
print("=" * 70)

print("\n  ── DIMENSION 1: REGRESSOR ARCHITECTURE ──")
print(f"  {'Option':<20} {'Spearman(pos)':<16} {'RMSE log':<12}"
      f"{'RevCap@1%':<12} {'RevCap@5%':<12} {'RevCap@10%':<12}")
print("  " + "-"*82)
for opt, r in reg_results.items():
    marker = " <<< WINNER" if opt == winner_reg else ""
    print(f"  {opt:<20} {r['Spearman pos val']:<16.4f} {r['RMSE log val']:<12.4f}"
          f"{r['RevCap@1% val(all)']:<12.1%} {r['RevCap@5% val(all)']:<12.1%}"
          f"{r['RevCap@10% val(all)']:<12.1%}{marker}")

print("\n  ── DIMENSION 2: SMEARING CORRECTION ──")
print(f"  {'Option':<28} {'MAE EUR':<14} {'Median AE':<14}"
      f"{'RevCap@1%':<12} {'RevCap@5%':<12}")
print("  " + "-"*80)
for opt, r in smear_results.items():
    marker = " <<< WINNER" if opt == smear_winner else ""
    print(f"  {opt:<28} {r['MAE EUR val']:<14,.0f} {r['Median AE EUR val']:<14,.0f}"
          f"{r['RevCap@1% pos val']:<12.1%} {r['RevCap@5% pos val']:<12.1%}{marker}")

print("\n  ── DIMENSION 3: CLASSIFIER TUNING ──")
print(f"  {'Config':<12} {'PR-AUC val':<14} {'ROC-AUC val':<14}"
      f"{'Recall@10%':<14} {'Trees'}")
print("  " + "-"*60)
for name, r in clf_results.items():
    marker = " <<< WINNER" if name == best_clf_name else ""
    print(f"  {name:<12} {r['PR-AUC val']:<14.4f} {r['ROC-AUC val']:<14.4f}"
          f"{r['Recall@10% val']:<14.4f} {r['Trees used']}{marker}")

print("\n  ── DIMENSION 4: FEATURE SET FOR REGRESSOR ──")
print(f"  {'Feature set':<28} {'Spearman(pos)':<16} {'RMSE log'}")
print("  " + "-"*54)
for name, r in feat_results.items():
    marker = " <<< WINNER" if name.startswith(feat_winner[:5]) else ""
    print(f"  {name:<28} {r['Spearman pos val']:<16.4f} {r['RMSE log val']:.4f}{marker}")

print("\n" + "=" * 70)
print("  RECOMMENDED CONFIG FOR M3 V2")
print("=" * 70)
print(f"  Classifier config      : {best_clf_name} — {best_clf_cfg}")
print(f"  Regressor architecture : {winner_reg}")
print(f"  Smearing correction    : {smear_winner}"
      + (f" (sigma^2={sigma2:.6f})" if smear_winner == "Log-normal smearing" else ""))
print(f"  Feature set            : {feat_winner}")
print(f"  CLV score formula      : CLV = P(active) * E[spend | active]")
if smear_winner == "Log-normal smearing" and winner_reg != "B-Tweedie":
    print(f"                           E[spend] = exp(reg_pred + {sigma2:.6f}/2)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Phase 8 — Save results to CSV
# ---------------------------------------------------------------------------
rows = []
for opt, r in reg_results.items():
    rows.append({"Dimension": "1-Regressor", "Option": opt, **r})
for opt, r in smear_results.items():
    rows.append({"Dimension": "2-Smearing", "Option": opt, **r})
for name, r in clf_results.items():
    rows.append({"Dimension": "3-Classifier", "Option": name, **r})
for name, r in feat_results.items():
    rows.append({"Dimension": "4-FeatureSet", "Option": name, **r})

rows.append({"Dimension": "WINNER", "Option": "Regressor Arch",
             "Winner": winner_reg})
rows.append({"Dimension": "WINNER", "Option": "Smearing",
             "Winner": smear_winner})
rows.append({"Dimension": "WINNER", "Option": "Classifier",
             "Winner": best_clf_name})
rows.append({"Dimension": "WINNER", "Option": "Feature Set",
             "Winner": feat_winner})

out_path = os.path.join(OUTPUT_DIR, "m3_comparison_results.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nResults saved to: output/tables/m3_comparison_results.csv")
print("\n=== M3 COMPARISON SCRIPT COMPLETE ===")
