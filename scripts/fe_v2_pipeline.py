"""
Feature Engineering V2 + Model Retraining — Cartier QTEM Data Challenge
========================================================================
Pipeline end-to-end autonoma. Non richiede intervento manuale.

Novità rispetto a v1:
  - Parsing colonne ALL_* da Aggregated_Data_clean (conteggi, diversità)
  - 5 feature di interazione (spend/seniority, trend×recency, ecc.)
  - CRC temporali per snapshot (filtro CREATION_DATE <= DATE_TARGET)
  - Classificatore XGBoost (non più LR) su tutto il train
  - Regressore XGBoost con validation fold snapshot 2015

Eseguire con:
  python scripts/fe_v2_pipeline.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
FEATURES_DIR  = os.path.join(ROOT, "data", "features")
OUTPUT_DIR    = os.path.join(ROOT, "output", "tables")
MODELS_DIR    = os.path.join(ROOT, "output", "models")

for d in [FEATURES_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

PROTECTED = [
    "CLIENT_ID", "DATE_TARGET",
    "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
    "LOG_TARGET_3Y", "LOG_TARGET_5Y",
    "BINARY_TARGET_3Y", "BINARY_TARGET_5Y",
]

# ---------------------------------------------------------------------------
# Fase 1 — Caricamento
# ---------------------------------------------------------------------------
def load_data():
    print("=" * 60)
    print("FASE 1 — CARICAMENTO")
    print("=" * 60)

    required = {
        "train": os.path.join(FEATURES_DIR, "train_features_final.csv"),
        "test":  os.path.join(FEATURES_DIR, "test_features_final.csv"),
        "agg":   os.path.join(PROCESSED_DIR, "Aggregated_Data_clean.csv"),
        "crc":   os.path.join(PROCESSED_DIR, "CRC_clean.csv"),
    }
    for name, path in required.items():
        if not os.path.exists(path):
            print(f"  ERRORE CRITICO: {path} non trovato.")
            sys.exit(1)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [OK] {name}: {size_mb:.0f} MB")

    train = pd.read_csv(required["train"], low_memory=False)
    test  = pd.read_csv(required["test"],  low_memory=False)
    train["DATE_TARGET"] = pd.to_datetime(train["DATE_TARGET"])
    test["DATE_TARGET"]  = pd.to_datetime(test["DATE_TARGET"])
    print(f"\n  Train v1: {train.shape}")
    print(f"  Test v1:  {test.shape}")

    print("\n  Caricamento Aggregated_Data_clean (lento)...")
    agg = pd.read_csv(required["agg"], parse_dates=["DATE_TARGET"],
                      low_memory=False)
    print(f"  Aggregated_Data_clean: {agg.shape}")

    crc = pd.read_csv(required["crc"], parse_dates=["CREATION_DATE"])
    print(f"  CRC_clean: {crc.shape}")
    print(f"  CRC ORIGIN values: {crc['ORIGIN'].value_counts().to_dict()}")

    all_cols = [c for c in agg.columns if c.startswith("ALL_")]
    print(f"\n  Colonne ALL_*: {all_cols}")

    return train, test, agg, crc, all_cols


# ---------------------------------------------------------------------------
# Fase 2 — Parsing ALL_* (conteggio e diversità)
# ---------------------------------------------------------------------------
def parse_all_columns(agg: pd.DataFrame, all_cols: list) -> pd.DataFrame:
    """
    Estrae feature aggregate dalle colonne ALL_*.
    Per ogni colonna produce: HAS_, N_TOTAL_, N_DISTINCT_, DIVERSITY_.
    Le colonne DATE vengono trattate separatamente con verifica anti-leakage.
    """
    print("\n" + "=" * 60)
    print("FASE 2 — PARSING ALL_* COLUMNS")
    print("=" * 60)

    result = agg[["CLIENT_ID", "DATE_TARGET"]].copy()
    date_cols    = [c for c in all_cols if "DATE" in c.upper()]
    non_date_cols = [c for c in all_cols if "DATE" not in c.upper()]

    def count_items(s: pd.Series):
        """Conta elementi in colonne CSV string."""
        filled = s.fillna("")
        return (
            filled.str.count(",") + 1
        ).where(filled.str.strip() != "", 0).astype(int)

    def count_distinct(s: pd.Series):
        """Conta elementi distinti in colonne CSV string."""
        def _n_distinct(val):
            if not val or str(val).strip() in ("", "nan"):
                return 0
            return len(set(x.strip() for x in str(val).split(",") if x.strip()))
        return s.fillna("").apply(_n_distinct)

    # Colonne non-DATE: feature complete
    for col in non_date_cols:
        n_total   = count_items(agg[col])
        n_distinct = count_distinct(agg[col])
        result[f"HAS_{col}"]       = (n_total > 0).astype(int)
        result[f"N_TOTAL_{col}"]   = n_total
        result[f"N_DISTINCT_{col}"] = n_distinct
        result[f"DIVERSITY_{col}"]  = np.where(n_total > 0,
                                                n_distinct / n_total, 0.0)
        print(f"  {col}: coverage={( n_total > 0).mean():.1%}")

    # Colonne DATE: verifica leakage su campione, poi solo N_TOTAL e N_DISTINCT
    # (le date stesse non vengono usate come feature per evitare leakage)
    for col in date_cols:
        n_total    = count_items(agg[col])
        n_distinct = count_distinct(agg[col])

        # Verifica anti-leakage su 200 righe casuali
        sample = agg[["DATE_TARGET", col]].dropna(subset=[col]).sample(
            min(200, agg[col].notna().sum()), random_state=42)
        violations = 0
        for _, row in sample.iterrows():
            cutoff = pd.Timestamp(row["DATE_TARGET"])
            for d in str(row[col]).split(","):
                d = d.strip()
                if not d:
                    continue
                try:
                    if pd.Timestamp(d) > cutoff:
                        violations += 1
                except Exception:
                    pass
        if violations > 0:
            print(f"  ATTENZIONE: {col} ha {violations} violazioni leakage — skip")
            continue

        result[f"N_TOTAL_{col}"]    = n_total
        result[f"N_DISTINCT_{col}"] = n_distinct
        print(f"  {col}: coverage={(n_total > 0).mean():.1%}, "
              f"leakage violations=0")

    new_cols = [c for c in result.columns if c not in ["CLIENT_ID", "DATE_TARGET"]]
    print(f"\n  Feature ALL_* create: {len(new_cols)}")
    assert len(new_cols) >= 6, f"Solo {len(new_cols)} feature ALL_* create"
    print(f"  [CHECK 2] PASS — {len(new_cols)} feature ALL_*")
    return result


# ---------------------------------------------------------------------------
# Fase 3 — Feature di interazione
# ---------------------------------------------------------------------------
def build_interactions(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """5 feature di interazione calcolate su df."""
    feat = df[["CLIENT_ID", "DATE_TARGET"]].copy()
    created = []

    if "TO_PAST_3Y" in df.columns and "SENIORITY" in df.columns:
        feat["SPEND_PER_SENIORITY"] = (
            df["TO_PAST_3Y"] / (df["SENIORITY"] + 1)
        ).clip(upper=df["TO_PAST_3Y"].quantile(0.99))
        created.append("SPEND_PER_SENIORITY")

    if "SPEND_TREND" in df.columns and "RECENCY_DAYS" in df.columns:
        max_r = df["RECENCY_DAYS"].quantile(0.99)
        recency_inv = 1.0 - (df["RECENCY_DAYS"].clip(upper=max_r) / (max_r + 1))
        feat["TREND_X_RECENCY"] = df["SPEND_TREND"].fillna(1.0) * recency_inv
        created.append("TREND_X_RECENCY")

    if ("MAX_ARTICLE_WORLD_PRICE" in df.columns
            and "FLAG_HE_RATIO" in df.columns):
        cap = df["MAX_ARTICLE_WORLD_PRICE"].quantile(0.99)
        feat["HE_PRICE_PROPENSITY"] = (
            df["MAX_ARTICLE_WORLD_PRICE"].clip(upper=cap) *
            df["FLAG_HE_RATIO"].fillna(0)
        )
        created.append("HE_PRICE_PROPENSITY")

    if "NB_TRS_FULL_HIST" in df.columns and "SENIORITY" in df.columns:
        cap = df["NB_TRS_FULL_HIST"].quantile(0.99)
        feat["TRS_FREQUENCY_ANNUAL"] = (
            df["NB_TRS_FULL_HIST"].clip(upper=cap) /
            (df["SENIORITY"] / 365.0 + 1)
        )
        created.append("TRS_FREQUENCY_ANNUAL")

    if "TO_PAST_3Y" in df.columns and "TO_FULL_HIST" in df.columns:
        feat["RECENT_HIST_RATIO"] = np.where(
            df["TO_FULL_HIST"] > 0,
            (df["TO_PAST_3Y"] / df["TO_FULL_HIST"]).clip(0, 1),
            0.0,
        )
        created.append("RECENT_HIST_RATIO")

    print(f"  {label}: {len(created)} feature create: {created}")
    assert len(created) >= 3
    return feat


# ---------------------------------------------------------------------------
# Fase 4 — CRC temporali per snapshot
# ---------------------------------------------------------------------------
def build_crc_snapshot_features(crc: pd.DataFrame,
                                 snapshots) -> pd.DataFrame:
    """
    Per ogni snapshot: conta CRC con CREATION_DATE <= DATE_TARGET.
    Anti-leakage: solo interazioni prima del cutoff.
    """
    print("\n" + "=" * 60)
    print("FASE 4 — CRC TEMPORALI PER SNAPSHOT")
    print("=" * 60)

    results = []
    for snap in sorted(snapshots):
        snap_ts = pd.Timestamp(snap)
        filtered = crc[crc["CREATION_DATE"] <= snap_ts].copy()

        agg_crc = filtered.groupby("CLIENT_ID").agg(
            N_CRC_SNAPSHOT       =("APPOINTMENT_ID", "count"),
        ).reset_index()

        # Flag Clienteling
        clienteling = (
            filtered[filtered["ORIGIN"] == "Clienteling"]
            .groupby("CLIENT_ID").size()
            .rename("HAS_CLIENTELING_SNAP")
            .gt(0).astype(int)
            .reset_index()
        )
        agg_crc = agg_crc.merge(clienteling, on="CLIENT_ID", how="left")
        agg_crc["HAS_CLIENTELING_SNAP"] = \
            agg_crc["HAS_CLIENTELING_SNAP"].fillna(0).astype(int)

        # Durata media (dove disponibile)
        if "APPOINTMENT_DURATION" in filtered.columns:
            dur = (filtered[filtered["APPOINTMENT_DURATION"].notna()]
                   .groupby("CLIENT_ID")["APPOINTMENT_DURATION"]
                   .mean().rename("AVG_DURATION_SNAP").reset_index())
            agg_crc = agg_crc.merge(dur, on="CLIENT_ID", how="left")

        agg_crc["HAS_CRC_SNAP"] = 1
        agg_crc["DATE_TARGET"]  = snap_ts
        results.append(agg_crc)
        print(f"  Snapshot {snap_ts.date()}: {len(agg_crc):,} clienti con CRC")

    df = pd.concat(results, ignore_index=True)
    print(f"  CRC features totali: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Fase 5 — Join master dataset v2
# ---------------------------------------------------------------------------
def build_v2(base: pd.DataFrame,
             all_feat: pd.DataFrame,
             int_feat: pd.DataFrame,
             crc_feat: pd.DataFrame,
             label: str) -> pd.DataFrame:
    """LEFT JOIN di tutte le nuove feature sul dataset base."""

    def align_dt(df):
        df = df.copy()
        df["DATE_TARGET"] = pd.to_datetime(df["DATE_TARGET"])
        return df

    base     = align_dt(base)
    all_feat = align_dt(all_feat)
    int_feat = align_dt(int_feat)
    crc_feat = align_dt(crc_feat)

    merged = base.copy()

    merged = merged.merge(all_feat, on=["CLIENT_ID", "DATE_TARGET"],
                          how="left", suffixes=("", "_dup"))
    merged = merged.merge(int_feat, on=["CLIENT_ID", "DATE_TARGET"],
                          how="left", suffixes=("", "_dup"))

    crc_new = [c for c in crc_feat.columns
               if c not in ["CLIENT_ID", "DATE_TARGET"]]
    merged = merged.merge(crc_feat, on=["CLIENT_ID", "DATE_TARGET"],
                          how="left", suffixes=("", "_dup"))
    for col in crc_new:
        if col in merged.columns:
            if col in ("HAS_CRC_SNAP", "N_CRC_SNAPSHOT", "HAS_CLIENTELING_SNAP"):
                merged[col] = merged[col].fillna(0)

    merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")],
                inplace=True, errors="ignore")

    print(f"  {label}: {merged.shape}  "
          f"(+{merged.shape[1] - base.shape[1]} colonne)")
    return merged


# ---------------------------------------------------------------------------
# Fase 6 — Feature selection v2
# ---------------------------------------------------------------------------
NEW_PREFIXES = (
    "HAS_ALL_", "N_TOTAL_ALL_", "N_DISTINCT_ALL_", "DIVERSITY_ALL_",
    "SPEND_PER_", "TREND_X_", "HE_PRICE_", "TRS_FREQ", "RECENT_HIST_",
    "N_CRC_", "HAS_CRC_", "AVG_DURATION_", "HAS_CLIENTELING_",
)


def select_features(train_v2: pd.DataFrame, test_v2: pd.DataFrame):
    """Near-zero variance + alta correlazione (r>0.98) — preserva nuove feature."""
    print("\n" + "=" * 60)
    print("FASE 6 — FEATURE SELECTION V2")
    print("=" * 60)

    feat_cols = [c for c in train_v2.columns if c not in PROTECTED]

    # Near-zero variance (>99% zero)
    pct_zero = (train_v2[feat_cols] == 0).mean()
    remove_nzv = {
        c for c in pct_zero[pct_zero > 0.99].index
        if c not in PROTECTED and
        not any(c.startswith(p) for p in NEW_PREFIXES)
    }

    # >99% null
    pct_null = train_v2[feat_cols].isnull().mean()
    remove_null = {
        c for c in pct_null[pct_null > 0.99].index
        if c not in PROTECTED and
        not any(c.startswith(p) for p in NEW_PREFIXES)
    }

    # Colonne costanti
    remove_const = {
        c for c in feat_cols
        if train_v2[c].nunique() <= 1 and c not in PROTECTED and
        not any(c.startswith(p) for p in NEW_PREFIXES)
    }

    all_remove = (remove_nzv | remove_null | remove_const) - set(PROTECTED)
    print(f"  Rimosse near-zero/null/const: {len(all_remove)}")

    # Alta correlazione (campione 30k) — non tocca nuove feature
    sample_cols = [c for c in feat_cols if c not in all_remove]
    # Solo colonne numeriche per la matrice di correlazione
    numeric_sample_cols = [c for c in sample_cols
                           if train_v2[c].dtype.kind in "iufb"]
    sample = (train_v2[numeric_sample_cols].sample(min(30_000, len(train_v2)),
                                                   random_state=42)
              .fillna(0))
    corr = sample.corr().abs()

    numeric_feat_cols = [c for c in feat_cols
                         if train_v2[c].dtype.kind in "iufb"]
    target_corr = (train_v2[numeric_feat_cols + ["TARGET_3Y"]].fillna(0)
                   .corr()["TARGET_3Y"])

    remove_corr = set()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    for col in upper.columns:
        if any(col.startswith(p) for p in NEW_PREFIXES) or col in PROTECTED:
            continue
        partners = upper[col][upper[col] > 0.98].index.tolist()
        for partner in partners:
            if (any(partner.startswith(p) for p in NEW_PREFIXES)
                    or partner in PROTECTED
                    or col in all_remove or partner in all_remove):
                continue
            if abs(target_corr.get(col, 0)) >= abs(target_corr.get(partner, 0)):
                remove_corr.add(partner)
            else:
                remove_corr.add(col)
    remove_corr -= set(PROTECTED)
    all_remove  |= remove_corr
    print(f"  Rimosse alta correlazione:    {len(remove_corr)}")

    train_f = train_v2.drop(columns=list(all_remove), errors="ignore")
    test_f  = test_v2.drop(columns=list(all_remove),  errors="ignore")

    assert list(train_f.columns) == list(test_f.columns), \
        "Colonne train/test non allineate dopo selezione"

    n_feat = len([c for c in train_f.columns if c not in PROTECTED])
    print(f"  Feature finali v2: {n_feat}")

    if n_feat < 75:
        print(f"  ATTENZIONE: {n_feat} < 75 — verifica parsing ALL_*")
    else:
        print(f"  [CHECK 5] Feature count >= 75: PASS")

    return train_f, test_f, n_feat, all_remove


# ---------------------------------------------------------------------------
# Fase 7 — Salvataggio
# ---------------------------------------------------------------------------
def save_datasets(train_f, test_f):
    print("\n" + "=" * 60)
    print("FASE 7 — SALVATAGGIO DATASET V2")
    print("=" * 60)

    tr_path = os.path.join(FEATURES_DIR, "train_features_final.csv")
    te_path = os.path.join(FEATURES_DIR, "test_features_final.csv")
    train_f.to_csv(tr_path, index=False)
    test_f.to_csv(te_path,  index=False)

    # Verifica reload
    check = pd.read_csv(tr_path, nrows=3)
    assert check.shape[1] == train_f.shape[1]
    print(f"  train_features_final.csv: {train_f.shape}  — OK")
    print(f"  test_features_final.csv:  {test_f.shape}  — OK")


# ---------------------------------------------------------------------------
# Fase 8 — Retraining modello
# ---------------------------------------------------------------------------
def retrain_model(train_f: pd.DataFrame, test_f: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FASE 8 — RETRAINING MODELLO V2")
    print("=" * 60)

    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 mean_squared_error, mean_absolute_error)
    import xgboost as xgb
    print(f"  XGBoost: {xgb.__version__}")

    from sklearn.preprocessing import OrdinalEncoder

    feat_cols = [c for c in train_f.columns if c not in PROTECTED]
    print(f"  Feature per il modello: {len(feat_cols)}")

    # Identifica colonne categoriali (object/string) e le encode
    cat_cols = [c for c in feat_cols if train_f[c].dtype == object]
    num_cols = [c for c in feat_cols if train_f[c].dtype != object]
    print(f"  Colonne categoriali: {cat_cols}")

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr_cat = enc.fit_transform(train_f[cat_cols].fillna("__missing__"))
        te_cat = enc.transform(test_f[cat_cols].fillna("__missing__"))
        joblib.dump(enc, os.path.join(MODELS_DIR, "ordinal_encoder_v2.pkl"))
        tr_df = pd.DataFrame(tr_cat, columns=cat_cols)
        te_df = pd.DataFrame(te_cat, columns=cat_cols)
        X_train_df = pd.concat([train_f[num_cols].reset_index(drop=True),
                                 tr_df], axis=1)
        X_test_df  = pd.concat([test_f[num_cols].reset_index(drop=True),
                                 te_df], axis=1)
    else:
        X_train_df = train_f[num_cols].reset_index(drop=True)
        X_test_df  = test_f[num_cols].reset_index(drop=True)

    # Salva l'ordine effettivo delle feature per feature importance
    model_feat_names = list(X_train_df.columns)
    print(f"  Feature effettive per il modello: {len(model_feat_names)}")

    y_train_bin = train_f["BINARY_TARGET_3Y"].values.astype(int)
    y_train_log = train_f["LOG_TARGET_3Y"].values
    y_test_bin  = test_f["BINARY_TARGET_3Y"].values.astype(int)
    y_test_log  = test_f["LOG_TARGET_3Y"].values
    y_test_raw  = test_f["TARGET_3Y"].values

    # Imputation mediana dal train
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_train_df)
    X_te_imp = imp.transform(X_test_df)
    joblib.dump(imp, os.path.join(MODELS_DIR, "imputer_v2.pkl"))

    # Split validation: snapshot 2015 per early stopping interno
    dates    = pd.to_datetime(train_f["DATE_TARGET"])
    val_mask = (dates.dt.year == 2015).values
    trn_mask = ~val_mask
    print(f"\n  Train rows:      {trn_mask.sum():,}")
    print(f"  Validation rows: {val_mask.sum():,} (snapshot 2015)")

    # ---- PARTE 1: XGBoost Classifier ----
    print("\n  --- PARTE 1: Classificatore XGBoost ---")
    spos = (1 - y_train_bin.mean()) / y_train_bin.mean()
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=spos,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr_imp[trn_mask], y_train_bin[trn_mask],
            eval_set=[(X_tr_imp[val_mask], y_train_bin[val_mask])],
            verbose=100)
    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier_xgb_v2.pkl"))

    y_prob  = clf.predict_proba(X_te_imp)[:, 1]
    pr_auc  = average_precision_score(y_test_bin, y_prob)
    roc_auc = roc_auc_score(y_test_bin, y_prob)
    bp      = y_test_bin.mean()
    n       = len(y_test_bin)
    recall_top10 = y_test_bin[np.argsort(y_prob)[::-1][: n // 10]].sum() \
                   / y_test_bin.sum()

    print(f"\n  PR-AUC:          {pr_auc:.4f}  (v1: 0.2691, base: {bp:.4f})")
    print(f"  ROC-AUC:         {roc_auc:.4f}  (v1: 0.8509)")
    print(f"  Recall top 10%:  {recall_top10:.1%}  (v1: 50.2%)")

    # ---- PARTE 2: XGBoost Regressor ----
    print("\n  --- PARTE 2: Regressore XGBoost (val=snapshot 2015) ---")
    pos_mask = y_train_bin == 1
    pos_dates = dates[pos_mask]
    val_pos   = (pos_dates.dt.year == 2015).values
    trn_pos   = ~val_pos

    X_tr_pos = X_tr_imp[pos_mask][trn_pos]
    y_tr_pos = y_train_log[pos_mask][trn_pos]
    X_vl_pos = X_tr_imp[pos_mask][val_pos]
    y_vl_pos = y_train_log[pos_mask][val_pos]

    print(f"  Train positivi: {len(X_tr_pos):,}  |  Val positivi: {len(X_vl_pos):,}")

    reg = xgb.XGBRegressor(
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
    reg.fit(X_tr_pos, y_tr_pos,
            eval_set=[(X_vl_pos, y_vl_pos)],
            verbose=100)
    joblib.dump(reg, os.path.join(MODELS_DIR, "regressor_xgb_v2.pkl"))

    pos_test   = y_test_bin == 1
    y_pred_log = reg.predict(X_te_imp[pos_test])
    y_pred_eur = np.expm1(y_pred_log)
    rmse_log   = float(np.sqrt(mean_squared_error(y_test_log[pos_test], y_pred_log)))
    mae_eur    = float(mean_absolute_error(y_test_raw[pos_test], y_pred_eur))
    median_ae  = float(np.median(np.abs(y_test_raw[pos_test] - y_pred_eur)))
    base_rmse  = float(np.sqrt(mean_squared_error(
        y_test_log[pos_test],
        np.full(pos_test.sum(), y_tr_pos.mean()))))

    print(f"\n  RMSE log-space:  {rmse_log:.4f}  (v1: 1.1346, base: {base_rmse:.4f})")
    print(f"  MAE EUR:         {mae_eur:,.0f}  (v1: 5.178)")
    print(f"  Median AE EUR:   {median_ae:,.0f}  (v1: 1.718)")

    # Revenue capture
    exp_spend = np.expm1(reg.predict(X_te_imp))
    combined  = y_prob * exp_spend

    res = pd.DataFrame({
        "CLIENT_ID":          test_f["CLIENT_ID"].values,
        "TARGET_3Y_actual":   y_test_raw,
        "COMBINED_PREDICTION": combined,
        "BINARY_actual":      y_test_bin,
        "P_SPEND":            y_prob,
    })
    res_sorted = res.sort_values("COMBINED_PREDICTION", ascending=False)
    total_rev  = y_test_raw.sum()

    rev_cap = {}
    print("\n  Revenue capture:")
    for pct, v1 in [(0.01, 0.206), (0.05, 0.470), (0.10, 0.602), (0.20, 0.743)]:
        n_top = max(1, int(len(res) * pct))
        rc = res_sorted.head(n_top)["TARGET_3Y_actual"].sum() / total_rev
        rev_cap[pct] = float(rc)
        print(f"    Top {pct:.0%}: {rc:.1%}  (v1: {v1:.1%})")

    res.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_v2.csv"), index=False)

    # Feature importance — usa get_score() per gestire feature con importance=0
    def booster_importance(model, feat_names, importance_type="gain"):
        scores = model.get_booster().get_score(importance_type=importance_type)
        rows = []
        for k, v in scores.items():
            # XGBoost usa "f0", "f1"... quando input e' numpy array
            try:
                idx = int(k.lstrip("f"))
                fname = feat_names[idx] if idx < len(feat_names) else k
            except (ValueError, IndexError):
                fname = k
            rows.append({"feature": fname, "importance": v})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    fi_clf = booster_importance(clf, model_feat_names)
    fi_reg = booster_importance(reg, model_feat_names)
    fi_clf.to_csv(os.path.join(OUTPUT_DIR, "clf_v2_importance.csv"), index=False)
    fi_reg.to_csv(os.path.join(OUTPUT_DIR, "reg_v2_importance.csv"), index=False)
    print("\n  Top 10 feature (classificatore v2):")
    print(fi_clf.head(10).to_string(index=False))

    return (pr_auc, roc_auc, recall_top10, bp,
            rmse_log, mae_eur, median_ae, base_rmse,
            rev_cap)


# ---------------------------------------------------------------------------
# Fase 9 — Definition of Done + Report
# ---------------------------------------------------------------------------
def validate_and_report(train_final, n_feat,
                        pr_auc, roc_auc, recall_top10,
                        rmse_log, mae_eur, median_ae,
                        rev_cap):
    print("\n" + "=" * 60)
    print("FASE 9 — DEFINITION OF DONE")
    print("=" * 60)

    new_all = [c for c in train_final.columns
               if any(c.startswith(p) for p in
                      ("HAS_ALL_", "N_TOTAL_ALL_", "N_DISTINCT_ALL_", "DIVERSITY_ALL_"))]
    new_int = [c for c in train_final.columns
               if any(c.startswith(p) for p in
                      ("SPEND_PER_", "TREND_X_", "HE_PRICE_", "TRS_FREQ", "RECENT_HIST_"))]
    new_crc = [c for c in train_final.columns
               if any(c.startswith(p) for p in
                      ("N_CRC_", "HAS_CRC_", "AVG_DURATION_", "HAS_CLIENTELING_"))]

    dod = [
        ("Feature count >= 75",          n_feat >= 75,           str(n_feat)),
        ("ALL_* features >= 6",          len(new_all) >= 6,      str(len(new_all))),
        ("Interaction features >= 3",    len(new_int) >= 3,      str(len(new_int))),
        ("CRC temporal features >= 1",   len(new_crc) >= 1,      str(len(new_crc))),
        ("PR-AUC >= 0.25",               pr_auc >= 0.25,         f"{pr_auc:.4f}"),
        ("ROC-AUC >= 0.80",              roc_auc >= 0.80,        f"{roc_auc:.4f}"),
        ("Revenue capture top 10% >= 55%", rev_cap[0.10] >= 0.55, f"{rev_cap[0.10]:.1%}"),
        ("train_features_final.csv OK",  os.path.exists(
             os.path.join(FEATURES_DIR, "train_features_final.csv")), ""),
        ("test_features_final.csv OK",   os.path.exists(
             os.path.join(FEATURES_DIR, "test_features_final.csv")), ""),
        ("Modelli v2 salvati",           os.path.exists(
             os.path.join(MODELS_DIR, "classifier_xgb_v2.pkl")), ""),
    ]

    all_pass = True
    for name, ok, val in dod:
        st = "PASS" if ok else "FAIL"
        print(f"  [{st}] {name}" + (f" -- {val}" if val else ""))
        if not ok:
            all_pass = False

    esito = ("COMPLETATO" if all_pass
             else "INCOMPLETO -- verificare i FAIL")
    print(f"\n  ESITO: {esito}")

    # Report CSV confronto v1 vs v2
    rows = [
        ("FE",      "Feature totali",          "63",   str(n_feat)),
        ("FE",      "Feature ALL_* parsate",   "0",    str(len(new_all))),
        ("FE",      "Feature interazione",     "0",    str(len(new_int))),
        ("FE",      "Feature CRC temporali",   "1 (HAS_CRC)",str(len(new_crc))),
        ("CLF",     "Algoritmo",               "LR-snap2018","XGBoost full train"),
        ("CLF",     "PR-AUC",                  "0.2691",f"{pr_auc:.4f}"),
        ("CLF",     "ROC-AUC",                 "0.8509",f"{roc_auc:.4f}"),
        ("CLF",     "Recall top decile",       "50.2%", f"{recall_top10:.1%}"),
        ("REG",     "RMSE log-space",          "1.1346",f"{rmse_log:.4f}"),
        ("REG",     "MAE EUR",                 "5178",  f"{mae_eur:.0f}"),
        ("REG",     "Median AE EUR",           "1718",  f"{median_ae:.0f}"),
        ("REV_CAP", "Top 1%",                  "20.6%", f"{rev_cap[0.01]:.1%}"),
        ("REV_CAP", "Top 5%",                  "47.0%", f"{rev_cap[0.05]:.1%}"),
        ("REV_CAP", "Top 10%",                 "60.2%", f"{rev_cap[0.10]:.1%}"),
        ("REV_CAP", "Top 20%",                 "74.3%", f"{rev_cap[0.20]:.1%}"),
    ]
    for name, ok, val in dod:
        rows.append(("DoD", name, "", "PASS" if ok else "FAIL"))

    pd.DataFrame(rows, columns=["categoria", "metrica", "v1", "v2"]).to_csv(
        os.path.join(OUTPUT_DIR, "fe_v2_model_report.csv"), index=False)

    print(f"\n  Salvato: output/tables/fe_v2_model_report.csv")
    print("=" * 60)
    return all_pass


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_all():
    print("\n" + "=" * 60)
    print(" CARTIER QTEM — FE V2 + MODEL RETRAINING PIPELINE ")
    print("=" * 60)

    # Fase 1
    train, test, agg, crc, all_cols = load_data()

    # Fase 2 — ALL_* parsing
    all_feat = parse_all_columns(agg, all_cols)

    # Fase 3 — Interazioni (calcolate su train e test separatamente)
    print("\n" + "=" * 60)
    print("FASE 3 — FEATURE DI INTERAZIONE")
    print("=" * 60)
    int_train = build_interactions(train, "train")
    int_test  = build_interactions(test,  "test")
    n_int = len([c for c in int_train.columns
                 if c not in ["CLIENT_ID", "DATE_TARGET"]])
    print(f"  [CHECK 3] PASS — {n_int} feature interazione")

    # Fase 4 — CRC temporali
    all_snaps = pd.to_datetime(
        list(train["DATE_TARGET"].unique()) +
        list(test["DATE_TARGET"].unique())
    )
    crc_feat = build_crc_snapshot_features(crc, all_snaps)
    crc_train = crc_feat[crc_feat["DATE_TARGET"].isin(train["DATE_TARGET"])].copy()
    crc_test  = crc_feat[crc_feat["DATE_TARGET"].isin(test["DATE_TARGET"])].copy()

    # Fase 5 — Join
    print("\n" + "=" * 60)
    print("FASE 5 — JOIN DATASET V2")
    print("=" * 60)
    train_v2 = build_v2(train, all_feat, int_train, crc_train, "train_v2")
    test_v2  = build_v2(test,  all_feat, int_test,  crc_test,  "test_v2")
    assert list(train_v2.columns) == list(test_v2.columns)

    # Fase 6 — Feature selection
    train_f, test_f, n_feat, removed = select_features(train_v2, test_v2)

    # Fase 7 — Salvataggio
    save_datasets(train_f, test_f)

    # Fase 8 — Retraining
    (pr_auc, roc_auc, recall_top10, bp,
     rmse_log, mae_eur, median_ae, base_rmse,
     rev_cap) = retrain_model(train_f, test_f)

    # Fase 9 — Validazione e report
    all_pass = validate_and_report(
        train_f, n_feat,
        pr_auc, roc_auc, recall_top10,
        rmse_log, mae_eur, median_ae,
        rev_cap)

    print("\n=== ESECUZIONE COMPLETATA ===")
    return all_pass


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
