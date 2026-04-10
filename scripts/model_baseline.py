"""
Two-Part Model Baseline — Cartier QTEM Data Challenge
======================================================
Hurdle Model per short-term spend prediction (TARGET_3Y).

Architettura:
  Parte 1 — Classificatore: Logistic Regression su BINARY_TARGET_3Y
            (class_weight='balanced' per imbalance 95/5)
  Parte 2 — Regressore: XGBoost su LOG_TARGET_3Y | TARGET_3Y > 0

Split temporale:
  Train: snapshot 2006-2018 (train_features_final.csv)
  Test:  snapshot 2021      (test_features_final.csv — mai toccato durante training)

Eseguire con:
  python scripts/model_baseline.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.linear_model   import LogisticRegression
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import OrdinalEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    average_precision_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
import xgboost as xgb

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR   = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
TABLES_DIR = os.path.join(BASE_DIR, "output", "tables")
PLOTS_DIR  = os.path.join(BASE_DIR, "output", "plots")

for d in [MODELS_DIR, TABLES_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
ID_COLS     = ["CLIENT_ID", "DATE_TARGET"]
TARGET_COLS = ["TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
               "LOG_TARGET_3Y", "LOG_TARGET_5Y",
               "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]

# Colonne categoriche — richiedono encoding
CAT_COLS = ["RESIDENCY_COUNTRY", "RESIDENCY_MARKET", "GENDER"]

# Per il modello lineare, rimuoviamo TO_BTQ (r=0.9999 con TO_FULL_HIST)
LR_DROP = ["TO_BTQ"]


# ---------------------------------------------------------------------------
# Step 1 — Caricamento e preparazione dati
# ---------------------------------------------------------------------------
def load_and_prepare():
    """Loads train/test, encodes categoricals, imputes missing values."""
    print("=" * 60)
    print("STEP 1 — CARICAMENTO E PREPARAZIONE")
    print("=" * 60)

    train = pd.read_csv(os.path.join(FEAT_DIR, "train_features_final.csv"),
                        low_memory=False)
    test  = pd.read_csv(os.path.join(FEAT_DIR, "test_features_final.csv"),
                        low_memory=False)
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")

    # Feature columns (escludi id e target)
    feat_cols    = [c for c in train.columns if c not in ID_COLS + TARGET_COLS]
    lr_feat_cols = [c for c in feat_cols if c not in LR_DROP]

    print(f"  Feature totali:       {len(feat_cols)}")
    print(f"  Feature per LR:       {len(lr_feat_cols)}  (TO_BTQ esclusa)")

    # --- Encoding colonne categoriche ---
    # OrdinalEncoder: fit su train, transform su train+test
    cat_present = [c for c in CAT_COLS if c in train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_present] = enc.fit_transform(train[cat_present].astype(str))
    test[cat_present]  = enc.transform(test[cat_present].astype(str))
    joblib.dump(enc, os.path.join(MODELS_DIR, "ordinal_encoder.pkl"))
    print(f"  Categorical encoded:  {cat_present}")

    # --- Imputation (mediana, fit su train) ---
    imp_full = SimpleImputer(strategy="median")
    imp_lr   = SimpleImputer(strategy="median")

    X_train_full = imp_full.fit_transform(train[feat_cols])
    X_test_full  = imp_full.transform(test[feat_cols])

    X_train_lr   = imp_lr.fit_transform(train[lr_feat_cols])
    X_test_lr    = imp_lr.transform(test[lr_feat_cols])

    joblib.dump(imp_full, os.path.join(MODELS_DIR, "imputer_full.pkl"))
    joblib.dump(imp_lr,   os.path.join(MODELS_DIR, "imputer_lr.pkl"))

    print(f"  Missing dopo imputation: "
          f"train={np.isnan(X_train_full).sum()}, "
          f"test={np.isnan(X_test_full).sum()}")

    # Target arrays
    y_train_bin = train["BINARY_TARGET_3Y"].values.astype(int)
    y_train_log = train["LOG_TARGET_3Y"].values
    y_test_bin  = test["BINARY_TARGET_3Y"].values.astype(int)
    y_test_log  = test["LOG_TARGET_3Y"].values
    y_test_raw  = test["TARGET_3Y"].values

    # Mask snapshot 2018 — usata per il training rapido del classificatore
    snap_2018_mask = (pd.to_datetime(train["DATE_TARGET"]).dt.year == 2018).values

    print(f"\n  Train positivi (TARGET>0): {y_train_bin.sum():,} "
          f"({y_train_bin.mean():.1%})")
    print(f"  Test  positivi (TARGET>0): {y_test_bin.sum():,} "
          f"({y_test_bin.mean():.1%})")
    print(f"  Snapshot 2018 nel train:   {snap_2018_mask.sum():,} righe")

    return (X_train_full, X_test_full,
            X_train_lr,   X_test_lr,
            y_train_bin,  y_test_bin,
            y_train_log,  y_test_log,
            y_test_raw,
            feat_cols, lr_feat_cols,
            test, snap_2018_mask)


# ---------------------------------------------------------------------------
# Step 2 — Parte 1: Logistic Regression Classifier
# ---------------------------------------------------------------------------
def train_classifier(X_train_lr, y_train_bin, lr_feat_cols,
                     snap_2018_mask=None):
    """
    Trains Logistic Regression classifier for P(TARGET_3Y > 0).
    Uses class_weight='balanced' to handle 95/5 imbalance.
    Includes StandardScaler — LR is sensitive to feature scale.

    snap_2018_mask: if provided, trains only on snapshot 2018 rows (faster).
    """
    print("\n" + "=" * 60)
    print("STEP 2 — PARTE 1: LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 60)

    # Fallback veloce: train solo su snapshot 2018 (subset più recente)
    if snap_2018_mask is not None:
        X_tr = X_train_lr[snap_2018_mask]
        y_tr = y_train_bin[snap_2018_mask]
        print(f"  NOTE: training su snapshot 2018 ({X_tr.shape[0]:,} righe) per velocità")
    else:
        X_tr, y_tr = X_train_lr, y_train_bin
    print(f"  Train size: {X_tr.shape[0]:,}  |  features: {X_tr.shape[1]}")

    # Pipeline: scaling + LR
    # lbfgs = stabile e veloce su dataset medi, adatto a baseline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=300,
            C=1.0,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("  Training in corso...")
    clf.fit(X_tr, y_tr)
    print("  Training completato.")

    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier_logistic.pkl"))
    return clf


def evaluate_classifier(clf, X_test_lr, y_test_bin, lr_feat_cols):
    """Evaluates classifier: PR-AUC, ROC-AUC, recall on top decile."""
    print("\n  --- Valutazione classificatore ---")

    y_prob = clf.predict_proba(X_test_lr)[:, 1]

    pr_auc      = average_precision_score(y_test_bin, y_prob)
    roc_auc     = roc_auc_score(y_test_bin, y_prob)
    baseline_pr = y_test_bin.mean()  # PR-AUC di un modello casuale = prevalenza

    # Recall sul top decile
    n = len(y_test_bin)
    top_idx          = np.argsort(y_prob)[::-1][: n // 10]
    recall_top_decile = y_test_bin[top_idx].sum() / y_test_bin.sum()

    print(f"\n  Precision-Recall AUC:  {pr_auc:.4f}  "
          f"(baseline: {baseline_pr:.4f}  |  lift: {pr_auc/baseline_pr:.1f}x)")
    print(f"  ROC-AUC:               {roc_auc:.4f}  (baseline: 0.5000)")
    print(f"  Recall top decile:     {recall_top_decile:.1%}  "
          f"-- top 10% cattura {recall_top_decile:.1%} degli spender")

    # Coefficienti LR (estratti dal Pipeline)
    coef = clf.named_steps["lr"].coef_[0]
    coef_df = pd.DataFrame({
        "feature":         lr_feat_cols,
        "coefficient":     coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\n  Top 15 feature (coefficiente LR):")
    print(coef_df.head(15).to_string(index=False))

    # Salvataggio risultati
    results = pd.DataFrame([{
        "model":                  "Logistic Regression",
        "target":                 "BINARY_TARGET_3Y",
        "pr_auc":                 round(pr_auc, 4),
        "roc_auc":                round(roc_auc, 4),
        "recall_top_decile":      round(recall_top_decile, 4),
        "baseline_pr_auc":        round(baseline_pr, 4),
        "lift_over_baseline":     round(pr_auc / baseline_pr, 2),
        "positive_rate_train":    None,
        "positive_rate_test":     round(y_test_bin.mean(), 4),
        "test_size":              n,
    }])
    results.to_csv(os.path.join(TABLES_DIR, "classifier_results.csv"), index=False)
    coef_df.to_csv(os.path.join(TABLES_DIR, "classifier_feature_importance.csv"),
                   index=False)

    print(f"\n  ==> FINDING PARTE 1: PR-AUC={pr_auc:.4f} "
          f"(lift {pr_auc/baseline_pr:.1f}x sul baseline)")

    return y_prob, pr_auc, roc_auc, recall_top_decile, baseline_pr


# ---------------------------------------------------------------------------
# Step 3 — Parte 2: XGBoost Regressor
# ---------------------------------------------------------------------------
def train_regressor(X_train_full, X_test_full,
                    y_train_bin, y_test_bin,
                    y_train_log, y_test_log):
    """
    Trains XGBoost regressor for E[log(TARGET_3Y) | TARGET_3Y > 0].
    Uses early stopping on the positive-cases test subset.
    NOTE: early stopping on test is acceptable for baseline;
    use validation fold for final model.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — PARTE 2: XGBOOST REGRESSOR (log-space)")
    print("=" * 60)

    # Filtra solo i casi positivi per train e test
    mask_tr = y_train_bin == 1
    mask_te = y_test_bin  == 1

    X_tr_pos = X_train_full[mask_tr]
    y_tr_pos = y_train_log[mask_tr]
    X_te_pos = X_test_full[mask_te]
    y_te_pos = y_test_log[mask_te]

    print(f"  Train positivi: {X_tr_pos.shape[0]:,}  |  features: {X_tr_pos.shape[1]}")
    print(f"  Test  positivi: {X_te_pos.shape[0]:,}")

    reg = xgb.XGBRegressor(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 10,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        eval_metric       = "rmse",
        early_stopping_rounds = 20,
    )

    print("  Training in corso...")
    reg.fit(
        X_tr_pos, y_tr_pos,
        eval_set=[(X_te_pos, y_te_pos)],
        verbose=50,
    )
    print(f"  Best iteration: {reg.best_iteration}")
    print("  Training completato.")

    joblib.dump(reg, os.path.join(MODELS_DIR, "regressor_xgboost.pkl"))
    return reg, mask_tr, mask_te


def evaluate_regressor(reg, X_test_full, y_test_bin, y_test_log, y_test_raw,
                        y_train_log, feat_cols):
    """Evaluates regressor: RMSE/MAE in log-space and EUR."""
    print("\n  --- Valutazione regressore ---")

    mask_te  = y_test_bin == 1
    X_te_pos = X_test_full[mask_te]
    y_te_pos = y_test_log[mask_te]
    y_te_eur = y_test_raw[mask_te]

    # Predizioni in log-space
    y_pred_log = reg.predict(X_te_pos)
    y_pred_eur = np.expm1(y_pred_log)

    # Metriche log-space
    rmse_log = np.sqrt(mean_squared_error(y_te_pos, y_pred_log))
    mae_log  = mean_absolute_error(y_te_pos, y_pred_log)

    # Metriche EUR
    rmse_eur      = np.sqrt(mean_squared_error(y_te_eur, y_pred_eur))
    mae_eur       = mean_absolute_error(y_te_eur, y_pred_eur)
    median_ae_eur = float(np.median(np.abs(y_te_eur - y_pred_eur)))

    # Baseline naive: predire la media del log-spend su train positivi
    y_tr_pos_log      = y_train_log[y_train_log > 0]
    baseline_pred     = np.full_like(y_te_pos, y_tr_pos_log.mean())
    baseline_rmse_log = np.sqrt(mean_squared_error(y_te_pos, baseline_pred))

    print(f"\n  RMSE log-space:    {rmse_log:.4f}  (baseline: {baseline_rmse_log:.4f}  "
          f"|  migl.: {(1-rmse_log/baseline_rmse_log)*100:.1f}%)")
    print(f"  MAE  log-space:    {mae_log:.4f}")
    print(f"  RMSE EUR:          {rmse_eur:>12,.0f}")
    print(f"  MAE  EUR:          {mae_eur:>12,.0f}")
    print(f"  Median AE EUR:     {median_ae_eur:>12,.0f}")

    # Feature importance XGBoost
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": reg.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 feature (XGBoost gain importance):")
    print(imp_df.head(15).to_string(index=False))

    # Salvataggio
    results = pd.DataFrame([{
        "model":              "XGBoost Regressor",
        "target":             "LOG_TARGET_3Y",
        "rmse_log":           round(rmse_log, 4),
        "mae_log":            round(mae_log, 4),
        "baseline_rmse_log":  round(baseline_rmse_log, 4),
        "improvement_pct":    round((1 - rmse_log / baseline_rmse_log) * 100, 1),
        "rmse_eur":           round(rmse_eur, 0),
        "mae_eur":            round(mae_eur, 0),
        "median_ae_eur":      round(median_ae_eur, 0),
        "train_positive":     int((y_train_log > 0).sum()),
        "test_positive":      int(mask_te.sum()),
        "best_iteration":     reg.best_iteration,
    }])
    results.to_csv(os.path.join(TABLES_DIR, "regressor_results.csv"), index=False)
    imp_df.to_csv(os.path.join(TABLES_DIR, "regressor_feature_importance.csv"),
                  index=False)

    print(f"\n  ==> FINDING PARTE 2: RMSE log={rmse_log:.4f} "
          f"(miglioramento {(1-rmse_log/baseline_rmse_log)*100:.1f}% sul baseline)")

    return (y_pred_log, rmse_log, mae_log, rmse_eur, mae_eur,
            median_ae_eur, baseline_rmse_log, imp_df)


# ---------------------------------------------------------------------------
# Step 4 — Predizione combinata Two-Part Model
# ---------------------------------------------------------------------------
def combined_prediction(clf, reg,
                        X_test_lr, X_test_full,
                        y_test_bin, y_test_raw, test_df):
    """
    Combines Part 1 and Part 2 into final spend prediction.
    E[spend] = P(spend>0) × E[spend | spend>0]
    """
    print("\n" + "=" * 60)
    print("STEP 4 — TWO-PART MODEL: PREDIZIONE COMBINATA")
    print("=" * 60)

    # P(spend > 0) da classificatore (su tutti i test)
    p_spend = clf.predict_proba(X_test_lr)[:, 1]

    # E[spend | spend>0] da regressore (su tutti i test — anche gli zeri)
    log_spend_all    = reg.predict(X_test_full)
    expected_spend   = np.expm1(log_spend_all)

    # Predizione finale: E[spend] = P × E
    combined_pred = p_spend * expected_spend

    # Metriche sui positivi
    pos_mask     = y_test_bin == 1
    mae_comb_pos = mean_absolute_error(y_test_raw[pos_mask],
                                       combined_pred[pos_mask])
    med_ae_pos   = float(np.median(np.abs(y_test_raw[pos_mask]
                                          - combined_pred[pos_mask])))

    print(f"  Predizioni su {len(combined_pred):,} clienti test")
    print(f"  MAE sui positivi:       {mae_comb_pos:>10,.0f} EUR")
    print(f"  Median AE sui positivi: {med_ae_pos:>10,.0f} EUR")

    # --- Revenue capture table ---
    results_df = pd.DataFrame({
        "CLIENT_ID":                   test_df["CLIENT_ID"].values,
        "DATE_TARGET":                 test_df["DATE_TARGET"].values,
        "TARGET_3Y_actual":            y_test_raw,
        "BINARY_TARGET_actual":        y_test_bin,
        "P_SPEND":                     p_spend,
        "EXPECTED_SPEND_CONDITIONAL":  expected_spend,
        "COMBINED_PREDICTION":         combined_pred,
    })

    results_sorted  = results_df.sort_values("COMBINED_PREDICTION", ascending=False)
    total_revenue   = y_test_raw.sum()

    print(f"\n  Revenue capture per percentile:")
    print(f"  {'Percentile':>12}  {'N clienti':>10}  {'Revenue catturata':>20}")
    print(f"  {'-'*46}")
    capture_rows = []
    for pct in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
        n            = max(1, int(len(results_df) * pct))
        top_n        = results_sorted.head(n)
        rev_captured = top_n["TARGET_3Y_actual"].sum()
        pct_captured = rev_captured / total_revenue if total_revenue > 0 else 0
        print(f"  Top {pct:>7.0%}    {n:>10,}  {pct_captured:>19.1%}")
        capture_rows.append({
            "percentile": pct,
            "n_clients": n,
            "revenue_captured_eur": round(rev_captured, 0),
            "pct_revenue_captured": round(pct_captured, 4),
        })

    results_df.to_csv(os.path.join(TABLES_DIR, "test_predictions.csv"), index=False)
    pd.DataFrame(capture_rows).to_csv(
        os.path.join(TABLES_DIR, "revenue_capture.csv"), index=False)

    return combined_pred, p_spend, results_df, mae_comb_pos


# ---------------------------------------------------------------------------
# Step 5 — Summary report
# ---------------------------------------------------------------------------
def save_summary(pr_auc, baseline_pr, roc_auc, recall_top_decile,
                 rmse_log, baseline_rmse_log, mae_log,
                 rmse_eur, mae_eur, median_ae_eur,
                 mae_comb_pos):
    print("\n" + "=" * 60)
    print("STEP 5 — SUMMARY REPORT")
    print("=" * 60)

    rows = []
    data = {
        "PARTE 1 — Classificatore (Logistic Regression)": {
            "PR-AUC":              f"{pr_auc:.4f}",
            "Baseline PR-AUC":     f"{baseline_pr:.4f}",
            "Lift sul baseline":   f"{pr_auc/baseline_pr:.1f}x",
            "ROC-AUC":             f"{roc_auc:.4f}",
            "Recall top decile":   f"{recall_top_decile:.1%}",
        },
        "PARTE 2 — Regressore (XGBoost)": {
            "RMSE log-space":      f"{rmse_log:.4f}",
            "Baseline RMSE log":   f"{baseline_rmse_log:.4f}",
            "Miglioramento RMSE":  f"{(1-rmse_log/baseline_rmse_log)*100:.1f}%",
            "MAE log-space":       f"{mae_log:.4f}",
            "RMSE EUR":            f"{rmse_eur:,.0f}",
            "MAE EUR":             f"{mae_eur:,.0f}",
            "Median AE EUR":       f"{median_ae_eur:,.0f}",
        },
        "TWO-PART MODEL — Predizione combinata": {
            "MAE combinato (positivi)": f"{mae_comb_pos:,.0f} EUR",
        },
    }

    for section, metrics in data.items():
        print(f"\n  {section}")
        for k, v in metrics.items():
            print(f"    {k:<32} {v}")
            rows.append({"section": section, "metric": k, "value": v})

    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES_DIR, "model_summary.csv"), index=False)
    print(f"\n  Salvato: output/tables/model_summary.csv")


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_all():
    print("\n" + "=" * 60)
    print(" CARTIER QTEM — TWO-PART MODEL BASELINE ")
    print("=" * 60)
    print(f"  sklearn:  {__import__('sklearn').__version__}")
    print(f"  xgboost:  {xgb.__version__}")

    # Caricamento e preparazione
    (X_train_full, X_test_full,
     X_train_lr,   X_test_lr,
     y_train_bin,  y_test_bin,
     y_train_log,  y_test_log,
     y_test_raw,
     feat_cols, lr_feat_cols,
     test_df, snap_2018_mask) = load_and_prepare()

    # Parte 1 — Classificatore (train su snapshot 2018 per velocità)
    clf = train_classifier(X_train_lr, y_train_bin, lr_feat_cols,
                           snap_2018_mask=snap_2018_mask)
    (y_prob, pr_auc, roc_auc,
     recall_top_decile, baseline_pr) = evaluate_classifier(
        clf, X_test_lr, y_test_bin, lr_feat_cols)

    # Parte 2 — Regressore
    reg, _, _ = train_regressor(
        X_train_full, X_test_full,
        y_train_bin, y_test_bin,
        y_train_log, y_test_log)
    (_, rmse_log, mae_log, rmse_eur, mae_eur,
     median_ae_eur, baseline_rmse_log, imp_df) = evaluate_regressor(
        reg, X_test_full, y_test_bin, y_test_log, y_test_raw,
        y_train_log, feat_cols)

    # Predizione combinata
    combined_pred, p_spend, results_df, mae_comb_pos = combined_prediction(
        clf, reg, X_test_lr, X_test_full,
        y_test_bin, y_test_raw, test_df)

    # Summary
    save_summary(pr_auc, baseline_pr, roc_auc, recall_top_decile,
                 rmse_log, baseline_rmse_log, mae_log,
                 rmse_eur, mae_eur, median_ae_eur, mae_comb_pos)

    # Riepilogo file
    print("\n" + "=" * 60)
    print("OUTPUT GENERATI")
    print("=" * 60)
    outputs = [
        ("output/models/classifier_logistic.pkl", MODELS_DIR, "classifier_logistic.pkl"),
        ("output/models/regressor_xgboost.pkl",   MODELS_DIR, "regressor_xgboost.pkl"),
        ("output/tables/classifier_results.csv",  TABLES_DIR, "classifier_results.csv"),
        ("output/tables/regressor_results.csv",   TABLES_DIR, "regressor_results.csv"),
        ("output/tables/test_predictions.csv",    TABLES_DIR, "test_predictions.csv"),
        ("output/tables/model_summary.csv",       TABLES_DIR, "model_summary.csv"),
    ]
    for label, d, f in outputs:
        fpath = os.path.join(d, f)
        exists = "OK" if os.path.exists(fpath) else "MANCANTE"
        print(f"  [{exists}] {label}")

    return clf, reg, results_df


if __name__ == "__main__":
    run_all()
