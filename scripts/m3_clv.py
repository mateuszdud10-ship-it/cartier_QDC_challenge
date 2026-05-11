"""
M3 — Customer Lifetime Value (CLV)
Cartier QTEM Data Challenge

Two-Part Model su TARGET_5Y: classificatore XGBoost + regressore XGBoost
Segmentazione 4 tier, early detection, confronto M2 vs M3.

Eseguire con:
  python scripts/m3_clv.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              mean_squared_error, mean_absolute_error)
import xgboost as xgb

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(ROOT, "data", "features")
MODELS_DIR   = os.path.join(ROOT, "output", "models")
OUTPUT_DIR   = os.path.join(ROOT, "output", "tables")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROTECTED = ["CLIENT_ID", "DATE_TARGET",
             "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
             "LOG_TARGET_3Y", "LOG_TARGET_5Y",
             "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]


# ---------------------------------------------------------------------------
# Utility: feature importance da booster (gestisce XGBoost 3.x)
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
# Setup
# ---------------------------------------------------------------------------
print("=" * 60)
print(" CARTIER QTEM — M3: CUSTOMER LIFETIME VALUE ")
print("=" * 60)
print(f"XGBoost: {xgb.__version__}")

train = pd.read_csv(os.path.join(FEATURES_DIR, "train_features_final.csv"),
                    low_memory=False)
test  = pd.read_csv(os.path.join(FEATURES_DIR, "test_features_final.csv"),
                    low_memory=False)
train["DATE_TARGET"] = pd.to_datetime(train["DATE_TARGET"])
test["DATE_TARGET"]  = pd.to_datetime(test["DATE_TARGET"])

print(f"\nTrain: {train.shape}")
print(f"Test:  {test.shape}")

assert "TARGET_5Y"        in train.columns
assert "LOG_TARGET_5Y"    in train.columns
assert "BINARY_TARGET_5Y" in train.columns

print(f"\nTARGET_5Y — train positivi: {(train['BINARY_TARGET_5Y']==1).sum():,} "
      f"({train['BINARY_TARGET_5Y'].mean():.1%})")
print(f"TARGET_5Y — test positivi:  {(test['BINARY_TARGET_5Y']==1).sum():,} "
      f"({test['BINARY_TARGET_5Y'].mean():.1%})")

FEAT_COLS = [c for c in train.columns if c not in PROTECTED]
print(f"\nFeature totali: {len(FEAT_COLS)}")

# ---------------------------------------------------------------------------
# Fase 1 — Preparazione dati
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 1 — PREPARAZIONE DATI")
print("=" * 60)

y_train_bin = train["BINARY_TARGET_5Y"].values
y_train_log = train["LOG_TARGET_5Y"].values
y_test_bin  = test["BINARY_TARGET_5Y"].values
y_test_log  = test["LOG_TARGET_5Y"].values
y_test_raw  = test["TARGET_5Y"].values
y_train_raw = train["TARGET_5Y"].values

# OrdinalEncoder per colonne categoriali
cat_cols = [c for c in FEAT_COLS if train[c].dtype == object]
num_cols = [c for c in FEAT_COLS if train[c].dtype != object]
print(f"  Colonne categoriali: {cat_cols}")

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
tr_cat = enc.fit_transform(train[cat_cols].fillna("__missing__"))
te_cat = enc.transform(test[cat_cols].fillna("__missing__"))
joblib.dump(enc, os.path.join(MODELS_DIR, "ordinal_encoder_clv.pkl"))

tr_cat_df = pd.DataFrame(tr_cat, columns=cat_cols)
te_cat_df = pd.DataFrame(te_cat, columns=cat_cols)
X_train_df = pd.concat([train[num_cols].reset_index(drop=True), tr_cat_df], axis=1)
X_test_df  = pd.concat([test[num_cols].reset_index(drop=True),  te_cat_df], axis=1)
model_feat_names = list(X_train_df.columns)

# Imputer
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_df)
X_test_imp  = imputer.transform(X_test_df)
joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer_clv.pkl"))

# Validation fold: snapshot 2015
train_dates = train["DATE_TARGET"]
val_mask = (train_dates.dt.year == 2015).values
trn_mask = ~val_mask

print(f"\n  Train fold:      {trn_mask.sum():,} righe")
print(f"  Validation fold: {val_mask.sum():,} righe (snapshot 2015)")
print(f"  Test:            {len(y_test_bin):,} righe (snapshot 2021)")
print(f"\n  [CHECK 1] Preparazione dati: PASS")

# ---------------------------------------------------------------------------
# Fase 2 — Classificatore CLV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 2 — CLASSIFICATORE CLV: P(TARGET_5Y > 0)")
print("=" * 60)

scale_pos = (1 - y_train_bin.mean()) / y_train_bin.mean()
print(f"  Positive rate train: {y_train_bin.mean():.2%}")
print(f"  scale_pos_weight:    {scale_pos:.1f}")

clf_clv = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)
clf_clv.fit(
    X_train_imp[trn_mask], y_train_bin[trn_mask],
    eval_set=[(X_train_imp[val_mask], y_train_bin[val_mask])],
    verbose=100,
)
joblib.dump(clf_clv, os.path.join(MODELS_DIR, "classifier_clv.pkl"))
print(f"  Alberi usati: {clf_clv.best_iteration}")

y_prob_clv      = clf_clv.predict_proba(X_test_imp)[:, 1]
pr_auc_clv      = average_precision_score(y_test_bin, y_prob_clv)
roc_auc_clv     = roc_auc_score(y_test_bin, y_prob_clv)
baseline_pr     = y_test_bin.mean()
n               = len(y_test_bin)
top_dec_idx     = np.argsort(y_prob_clv)[::-1][: n // 10]
recall_top10_clv = y_test_bin[top_dec_idx].sum() / y_test_bin.sum()

print(f"\n  PR-AUC:            {pr_auc_clv:.4f}  (baseline: {baseline_pr:.4f})")
print(f"  ROC-AUC:           {roc_auc_clv:.4f}")
print(f"  Recall top decile: {recall_top10_clv:.4f}")
print(f"  Lift:              {pr_auc_clv/baseline_pr:.1f}x")

assert pr_auc_clv >= 0.20, f"PR-AUC {pr_auc_clv:.4f} < 0.20"
print(f"\n  [CHECK 2] Classificatore CLV: PASS")

# ---------------------------------------------------------------------------
# Fase 3 — Regressore CLV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 3 — REGRESSORE CLV: E[log(TARGET_5Y) | TARGET_5Y > 0]")
print("=" * 60)

pos_mask_tr = y_train_bin == 1
pos_mask_te = y_test_bin  == 1

pos_dates = train_dates[pos_mask_tr]
val_pos   = (pos_dates.dt.year == 2015).values
trn_pos   = ~val_pos

X_tr_pos = X_train_imp[pos_mask_tr][trn_pos]
y_tr_pos = y_train_log[pos_mask_tr][trn_pos]
X_vl_pos = X_train_imp[pos_mask_tr][val_pos]
y_vl_pos = y_train_log[pos_mask_tr][val_pos]

print(f"  Train positivi: {len(X_tr_pos):,}")
print(f"  Val positivi:   {len(X_vl_pos):,}")
print(f"  Test positivi:  {pos_mask_te.sum():,}")

reg_clv = xgb.XGBRegressor(
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
reg_clv.fit(X_tr_pos, y_tr_pos,
            eval_set=[(X_vl_pos, y_vl_pos)],
            verbose=100)
joblib.dump(reg_clv, os.path.join(MODELS_DIR, "regressor_clv.pkl"))
print(f"  Alberi usati: {reg_clv.best_iteration}")

y_pred_log_clv  = reg_clv.predict(X_test_imp[pos_mask_te])
y_pred_eur_clv  = np.expm1(y_pred_log_clv)
y_test_eur_pos  = y_test_raw[pos_mask_te]
rmse_log_clv    = np.sqrt(mean_squared_error(y_test_log[pos_mask_te], y_pred_log_clv))
mae_eur_clv     = mean_absolute_error(y_test_eur_pos, y_pred_eur_clv)
med_ae_clv      = float(np.median(np.abs(y_test_eur_pos - y_pred_eur_clv)))
spearman_pos, _ = spearmanr(y_test_eur_pos, y_pred_eur_clv)

baseline_rmse = np.sqrt(mean_squared_error(
    y_test_log[pos_mask_te],
    np.full(pos_mask_te.sum(), y_tr_pos.mean())))

print(f"\n  RMSE log-space: {rmse_log_clv:.4f}  (baseline: {baseline_rmse:.4f})")
print(f"  MAE EUR:        {mae_eur_clv:,.0f}")
print(f"  Median AE EUR:  {med_ae_clv:,.0f}")
print(f"  Spearman r:     {spearman_pos:.4f}")

assert spearman_pos >= 0.35, f"Spearman {spearman_pos:.4f} < 0.35"
print(f"\n  [CHECK 3] Regressore CLV: PASS")

# ---------------------------------------------------------------------------
# Fase 4 — Predizione combinata e revenue capture
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 4 — PREDIZIONE COMBINATA + REVENUE CAPTURE")
print("=" * 60)

p_spend_clv   = clf_clv.predict_proba(X_test_imp)[:, 1]
exp_spend_clv = np.expm1(reg_clv.predict(X_test_imp))
clv_score     = p_spend_clv * exp_spend_clv

res = pd.DataFrame({
    "CLIENT_ID":          test["CLIENT_ID"].values,
    "DATE_TARGET":        test["DATE_TARGET"].values,
    "TARGET_5Y_actual":   y_test_raw,
    "CLV_SCORE":          clv_score,
    "P_SPEND_5Y":         p_spend_clv,
    "EXP_SPEND_5Y":       exp_spend_clv,
    "BINARY_5Y_actual":   y_test_bin,
})

res_sorted  = res.sort_values("CLV_SCORE", ascending=False)
total_rev5y = res["TARGET_5Y_actual"].sum()

rev_cap = {}
print("  Revenue capture TARGET_5Y:")
for pct in [0.01, 0.05, 0.10, 0.20]:
    n_top = max(1, int(len(res) * pct))
    rc    = res_sorted.head(n_top)["TARGET_5Y_actual"].sum() / total_rev5y
    rev_cap[pct] = float(rc)
    print(f"    Top {pct:.0%}: {rc:.1%}")

assert rev_cap[0.10] >= 0.55, f"Revenue capture top 10% {rev_cap[0.10]:.1%} < 55%"
print(f"\n  [CHECK 4] Revenue capture: PASS")

# ---------------------------------------------------------------------------
# Fase 5 — Segmentazione 4 tier (snapshot 2021)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 5 — SEGMENTAZIONE CLV 4 TIER (snapshot 2021)")
print("=" * 60)

p2  = float(np.percentile(clv_score, 98))
p10 = float(np.percentile(clv_score, 90))
p40 = float(np.percentile(clv_score, 60))

def assign_tier(score):
    if   score >= p2:  return "VIC"
    elif score >= p10: return "High Spender"
    elif score >= p40: return "Aspirational"
    else:              return "Dormant"

res["CLV_TIER"] = res["CLV_SCORE"].apply(assign_tier)

tier_order = ["VIC", "High Spender", "Aspirational", "Dormant"]
tier_stats = res.groupby("CLV_TIER").agg(
    N_clienti         = ("CLIENT_ID",         "count"),
    CLV_score_medio   = ("CLV_SCORE",          "mean"),
    Spend_reale_medio = ("TARGET_5Y_actual",   "mean"),
    Pct_spender_reali = ("BINARY_5Y_actual",   "mean"),
    Revenue_catturata = ("TARGET_5Y_actual",   "sum"),
).reindex(tier_order)

tier_stats["Pct_revenue"] = (tier_stats["Revenue_catturata"] / total_rev5y * 100).round(1)
tier_stats["Pct_clienti"] = (tier_stats["N_clienti"] / len(res) * 100).round(1)

print("\n  Segmentazione CLV 4 tier:")
print(tier_stats[["N_clienti","Pct_clienti","Pct_revenue",
                   "Spend_reale_medio","Pct_spender_reali"]].to_string())
print(f"\n  Soglie CLV score:")
print(f"    VIC:           >= {p2:.2f}  (top 2%)")
print(f"    High Spender:  >= {p10:.2f}  (top 10%)")
print(f"    Aspirational:  >= {p40:.2f}  (top 40%)")

res.to_csv(os.path.join(OUTPUT_DIR, "clv_segmentation_2021.csv"), index=False)
tier_stats.to_csv(os.path.join(OUTPUT_DIR, "clv_tier_summary_2021.csv"))
print(f"\n  [CHECK 5] Segmentazione 4 tier: PASS")

# ---------------------------------------------------------------------------
# Fase 6 — CLV score su tutto il panel
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 6 — CLV SCORE PANEL COMPLETO (tutti gli snapshot)")
print("=" * 60)

full_panel = pd.concat([train, test], ignore_index=True)
full_panel["DATE_TARGET"] = pd.to_datetime(full_panel["DATE_TARGET"])

fp_cat = enc.transform(full_panel[cat_cols].fillna("__missing__"))
fp_cat_df = pd.DataFrame(fp_cat, columns=cat_cols)
X_full_df = pd.concat([full_panel[num_cols].reset_index(drop=True), fp_cat_df], axis=1)
X_full_imp = imputer.transform(X_full_df)

p_spend_full   = clf_clv.predict_proba(X_full_imp)[:, 1]
exp_spend_full = np.expm1(reg_clv.predict(X_full_imp))
clv_full       = p_spend_full * exp_spend_full

panel_out = pd.DataFrame({
    "CLIENT_ID":         full_panel["CLIENT_ID"].values,
    "DATE_TARGET":       full_panel["DATE_TARGET"].values,
    "CLV_SCORE":         clv_full,
    "P_SPEND_5Y":        p_spend_full,
    "EXP_SPEND_5Y":      exp_spend_full,
    "TARGET_5Y_actual":  full_panel["TARGET_5Y"].values,
    "BINARY_5Y_actual":  full_panel["BINARY_TARGET_5Y"].values,
})
panel_out["CLV_TIER"] = panel_out["CLV_SCORE"].apply(assign_tier)

print("\n  Distribuzione tier per snapshot:")
tier_by_snap = (panel_out.groupby(["DATE_TARGET","CLV_TIER"])
                .size().unstack(fill_value=0))
print(tier_by_snap.to_string())

panel_out.to_csv(os.path.join(OUTPUT_DIR, "clv_panel_all_snapshots.csv"), index=False)
print(f"\n  [CHECK 6] CLV panel completo: PASS")

# ---------------------------------------------------------------------------
# Fase 7 — Early detection analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 7 — EARLY DETECTION ANALYSIS")
print("=" * 60)

vic_2021 = set(res[res["CLV_TIER"] == "VIC"]["CLIENT_ID"].values)
hs_2021  = set(res[res["CLV_TIER"] == "High Spender"]["CLIENT_ID"].values)

print(f"  VIC nel 2021:          {len(vic_2021):,} clienti")
print(f"  High Spender nel 2021: {len(hs_2021):,} clienti")

snapshots_hist = sorted(panel_out["DATE_TARGET"].unique())
snapshots_hist = [s for s in snapshots_hist if pd.Timestamp(s).year < 2021]

print(f"\n  Tracciamento in {len(snapshots_hist)} snapshot storici:")

early_detection = []
for snap in snapshots_hist:
    snap_data = panel_out[panel_out["DATE_TARGET"] == snap]
    for group_name, group_ids in [("VIC_2021", vic_2021),
                                   ("HighSpender_2021", hs_2021)]:
        snap_group = snap_data[snap_data["CLIENT_ID"].isin(group_ids)]
        if len(snap_group) == 0:
            continue
        td = snap_group["CLV_TIER"].value_counts(normalize=True)
        early_detection.append({
            "snapshot":              snap,
            "group":                 group_name,
            "n_clienti_presenti":    len(snap_group),
            "pct_VIC":               td.get("VIC", 0),
            "pct_HighSpender":       td.get("High Spender", 0),
            "pct_Aspirational":      td.get("Aspirational", 0),
            "pct_Dormant":           td.get("Dormant", 0),
            "clv_score_medio":       snap_group["CLV_SCORE"].mean(),
        })

early_df = pd.DataFrame(early_detection)
if len(early_df):
    print(early_df.to_string(index=False))
    vic_traj = early_df[early_df["group"] == "VIC_2021"].copy()
    if len(vic_traj):
        vic_traj["pct_already_top"] = vic_traj["pct_VIC"] + vic_traj["pct_HighSpender"]
        print("\n  VIC 2021 -- % gia in VIC o High Spender negli snapshot storici:")
        for _, row in vic_traj.iterrows():
            print(f"    {str(row['snapshot'])[:10]}: "
                  f"{row['pct_already_top']:.1%} gia in top tier "
                  f"({row['n_clienti_presenti']:,} clienti presenti)")

early_df.to_csv(os.path.join(OUTPUT_DIR, "clv_early_detection.csv"), index=False)
print(f"\n  [CHECK 7] Early detection: PASS")

# ---------------------------------------------------------------------------
# Fase 8 — Feature importance CLV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 8 — FEATURE IMPORTANCE CLV")
print("=" * 60)

clf_imp_df = booster_importance(clf_clv, model_feat_names)
reg_imp_df = booster_importance(reg_clv, model_feat_names)

print("  Top 10 feature — Classificatore CLV:")
print(clf_imp_df.head(10).to_string(index=False))
print("\n  Top 10 feature — Regressore CLV:")
print(reg_imp_df.head(10).to_string(index=False))

clf_imp_df.to_csv(os.path.join(OUTPUT_DIR, "clv_clf_feature_importance.csv"), index=False)
reg_imp_df.to_csv(os.path.join(OUTPUT_DIR, "clv_reg_feature_importance.csv"), index=False)

top5_m2  = ["TO_PAST_3Y","TO_FULL_HIST","TO_BTQ","NB_TRS_FULL_HIST",
             "MAX_ARTICLE_WORLD_PRICE"]
top5_clv = clf_imp_df.head(5)["feature"].tolist()
overlap  = len(set(top5_m2) & set(top5_clv))
print(f"\n  Overlap top-5 feature M2 vs CLV: {overlap}/5")

# ---------------------------------------------------------------------------
# Fase 9 — Confronto M2 vs M3
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 9 — CONFRONTO M2 vs M3")
print("=" * 60)

m2_path = os.path.join(OUTPUT_DIR, "test_predictions_v2.csv")
if os.path.exists(m2_path):
    m2_preds   = pd.read_csv(m2_path)
    comparison = res.merge(
        m2_preds[["CLIENT_ID", "COMBINED_PREDICTION"]].rename(
            columns={"COMBINED_PREDICTION": "M2_SCORE"}),
        on="CLIENT_ID", how="inner")
    corr_m2_m3, _ = spearmanr(comparison["M2_SCORE"], comparison["CLV_SCORE"])
    print(f"  Correlazione Spearman M2 vs M3 score: {corr_m2_m3:.4f}")

    m2_top = set(comparison.nlargest(max(1, int(len(comparison)*0.10)), "M2_SCORE")["CLIENT_ID"])
    m3_top = set(comparison.nlargest(max(1, int(len(comparison)*0.10)), "CLV_SCORE")["CLIENT_ID"])
    overlap_top10 = len(m2_top & m3_top) / max(1, len(m2_top))
    print(f"  Overlap top 10% M2 vs M3: {overlap_top10:.1%}")
else:
    corr_m2_m3    = None
    overlap_top10 = None
    print("  test_predictions_v2.csv non trovato — confronto saltato")

# ---------------------------------------------------------------------------
# Fase 10 — Validazione finale e report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 10 — DEFINITION OF DONE")
print("=" * 60)

dod = [
    ("Classificatore CLV addestrato",
     os.path.exists(os.path.join(MODELS_DIR, "classifier_clv.pkl")), ""),
    ("PR-AUC >= 0.20",
     pr_auc_clv >= 0.20, f"{pr_auc_clv:.4f}"),
    ("Spearman regressore >= 0.35",
     spearman_pos >= 0.35, f"{spearman_pos:.4f}"),
    ("Revenue capture top 10% >= 55%",
     rev_cap[0.10] >= 0.55, f"{rev_cap[0.10]:.1%}"),
    ("Segmentazione 2021 salvata",
     os.path.exists(os.path.join(OUTPUT_DIR, "clv_segmentation_2021.csv")), ""),
    ("Panel completo salvato",
     os.path.exists(os.path.join(OUTPUT_DIR, "clv_panel_all_snapshots.csv")), ""),
    ("Early detection salvata",
     os.path.exists(os.path.join(OUTPUT_DIR, "clv_early_detection.csv")), ""),
]

all_pass = True
for name, ok, val in dod:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {val}" if val else ""))
    if not ok:
        all_pass = False

esito = "M3 COMPLETATO" if all_pass else "M3 INCOMPLETO"
print(f"\n  ESITO: {esito}")

# Report CSV
rows = [
    ("Modello",           "Algoritmo classificatore",  "XGBoost",              ""),
    ("Modello",           "PR-AUC",                    f"{pr_auc_clv:.4f}",    ""),
    ("Modello",           "ROC-AUC",                   f"{roc_auc_clv:.4f}",   ""),
    ("Modello",           "Recall top decile",          f"{recall_top10_clv:.1%}", ""),
    ("Modello",           "Spearman regressore",        f"{spearman_pos:.4f}",  ""),
    ("Modello",           "RMSE log-space",             f"{rmse_log_clv:.4f}",  ""),
    ("Modello",           "MAE EUR",                    f"{mae_eur_clv:,.0f}",  ""),
    ("Revenue Capture",   "Top 1%",                    f"{rev_cap[0.01]:.1%}", ""),
    ("Revenue Capture",   "Top 5%",                    f"{rev_cap[0.05]:.1%}", ""),
    ("Revenue Capture",   "Top 10%",                   f"{rev_cap[0.10]:.1%}", ""),
    ("Revenue Capture",   "Top 20%",                   f"{rev_cap[0.20]:.1%}", ""),
    ("Segmentazione",     "N tier",                    "4",
     "VIC / High Spender / Aspirational / Dormant"),
    ("Segmentazione",     "VIC (top 2%)",
     str(int(tier_stats.loc["VIC",       "N_clienti"])),
     f"{tier_stats.loc['VIC','Pct_revenue']:.1f}% revenue"),
    ("Segmentazione",     "High Spender (top 10%)",
     str(int(tier_stats.loc["High Spender", "N_clienti"])),
     f"{tier_stats.loc['High Spender','Pct_revenue']:.1f}% revenue"),
    ("Segmentazione",     "Aspirational (top 40%)",
     str(int(tier_stats.loc["Aspirational", "N_clienti"])),
     f"{tier_stats.loc['Aspirational','Pct_revenue']:.1f}% revenue"),
    ("Segmentazione",     "Dormant (bottom 60%)",
     str(int(tier_stats.loc["Dormant",    "N_clienti"])),
     f"{tier_stats.loc['Dormant','Pct_revenue']:.1f}% revenue"),
]
if corr_m2_m3 is not None:
    rows.append(("Confronto M2-M3", "Spearman M2 vs M3",      f"{corr_m2_m3:.4f}",    ""))
    rows.append(("Confronto M2-M3", "Overlap top 10% M2 vs M3", f"{overlap_top10:.1%}", ""))
for name, ok, val in dod:
    rows.append(("DoD", name, "PASS" if ok else "FAIL", val))

pd.DataFrame(rows, columns=["categoria", "metrica", "valore", "note"]
    ).to_csv(os.path.join(OUTPUT_DIR, "m3_clv_report.csv"), index=False)

print(f"\nFile prodotti:")
for f in ["clv_segmentation_2021.csv", "clv_tier_summary_2021.csv",
          "clv_panel_all_snapshots.csv", "clv_early_detection.csv",
          "clv_clf_feature_importance.csv", "clv_reg_feature_importance.csv",
          "m3_clv_report.csv"]:
    print(f"  output/tables/{f}")
for f in ["classifier_clv.pkl", "regressor_clv.pkl", "imputer_clv.pkl",
          "ordinal_encoder_clv.pkl"]:
    print(f"  output/models/{f}")

print(f"\n=== {esito} ===")
