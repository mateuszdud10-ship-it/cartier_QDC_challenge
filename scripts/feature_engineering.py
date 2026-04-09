"""
Feature Engineering — Cartier QTEM Data Challenge
==================================================
Costruisce il feature set completo per il Two-Part Model (Hurdle Model).
Segue le decisioni documentate in CLAUDE.md.

Struttura output:
  data/features/transaction_features.csv       — feature RFM da Transactions
  data/features/article_features.csv           — feature da Articles (join LEFT)
  data/features/aggregated_features.csv        — colonne selezionate da Aggregated_Data
  data/features/master_features_all_snapshots.csv — join completo
  data/features/master_features_with_targets.csv  — master + log-transform target
  data/features/train_features.csv             — snapshot 2006-2018 (train)
  data/features/test_features.csv              — snapshot 2021 (test, isolato)

Eseguire con:
  python scripts/feature_engineering.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
TABLES_DIR   = os.path.join(BASE_DIR, "output", "tables")

# Articles.csv non è in data/raw/ del repo — cerca in percorsi alternativi
ARTICLES_PATHS = [
    os.path.join(BASE_DIR, "data", "raw", "Articles.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "Articles.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "cartier_QDC_challenge", "data", "raw", "Articles.csv"),
]

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,   exist_ok=True)


# ---------------------------------------------------------------------------
# Fase 1 — Caricamento dati
# ---------------------------------------------------------------------------
def load_data():
    """Carica tutti i dataset processati necessari per il feature engineering."""
    print("=" * 60)
    print("FASE 1 — CARICAMENTO DATI")
    print("=" * 60)

    agg = pd.read_csv(
        os.path.join(PROCESSED, "Aggregated_Data_clean.csv"),
        parse_dates=["DATE_TARGET"],
        low_memory=False,
    )
    print(f"Aggregated_Data_clean: {agg.shape}")

    trans = pd.read_csv(
        os.path.join(PROCESSED, "Transactions_clean.csv"),
        parse_dates=["TRS_DATE"],
        low_memory=False,
    )
    print(f"Transactions_clean:    {trans.shape}")

    supp = pd.read_csv(
        os.path.join(PROCESSED, "supplementary_features.csv"),
        low_memory=False,
    )
    print(f"supplementary_features: {supp.shape}")

    # Articles — tenta percorsi alternativi
    articles = None
    for path in ARTICLES_PATHS:
        if os.path.exists(path):
            articles = pd.read_csv(path)
            print(f"Articles:              {articles.shape}  (da {path})")
            break
    if articles is None:
        print("ATTENZIONE: Articles.csv non trovato — feature articoli saltate.")

    snapshots = sorted(agg["DATE_TARGET"].unique())
    train_snaps = [s for s in snapshots if pd.Timestamp(s).year < 2021]
    test_snap   = pd.Timestamp("2021-01-01")

    print(f"\nSnapshot totali:  {len(snapshots)}: {[str(s)[:10] for s in snapshots]}")
    print(f"Snapshot train:   {len(train_snaps)}")
    print(f"Snapshot test:    {test_snap.date()}")

    return agg, trans, articles, supp, snapshots, train_snaps, test_snap


# ---------------------------------------------------------------------------
# Fase 2 — Feature da Transactions (RFM per snapshot)
# ---------------------------------------------------------------------------
def build_transaction_features(trans: pd.DataFrame,
                                date_target: pd.Timestamp) -> pd.DataFrame:
    """
    Builds RFM and behavioral features from transactions for a given snapshot.
    Anti-leakage: only uses transactions with TRS_DATE <= date_target.

    Returns: DataFrame with one row per CLIENT_ID.
    """
    # Filtro anti-leakage — CRITICO
    t = trans[trans["TRS_DATE"] <= date_target].copy()

    if len(t) == 0:
        return pd.DataFrame(columns=["CLIENT_ID", "DATE_TARGET"])

    # Separa Sales e Repairs
    sales   = t[t["TRS_CATEG"] == "Sale"]
    repairs = t[t["TRS_CATEG"] == "Repair"]

    # --- Feature aggregate di base ---
    feat_base = t.groupby("CLIENT_ID").agg(
        N_TRANSACTIONS      =("TRS_DATE",                "count"),
        TOTAL_SPEND         =("TO_WITHOUTTAX_EUR_CONST", "sum"),
        AVG_SPEND_PER_TRS   =("TO_WITHOUTTAX_EUR_CONST", "mean"),
        MAX_SINGLE_SPEND    =("TO_WITHOUTTAX_EUR_CONST", "max"),
        N_DISTINCT_ARTICLES =("ARTICLE_ID",              "nunique"),
        LAST_PURCHASE_DATE  =("TRS_DATE",                "max"),
        FIRST_PURCHASE_DATE =("TRS_DATE",                "min"),
    ).reset_index()

    # Recency e Tenure
    feat_base["RECENCY_DAYS"] = (
        date_target - feat_base["LAST_PURCHASE_DATE"]
    ).dt.days
    feat_base["TENURE_DAYS"] = (
        feat_base["LAST_PURCHASE_DATE"] - feat_base["FIRST_PURCHASE_DATE"]
    ).dt.days

    # --- Spend per finestre temporali ---
    cutoff_3y   = date_target - pd.DateOffset(years=3)
    cutoff_6y   = date_target - pd.DateOffset(years=6)

    spend_3y = (
        t[t["TRS_DATE"] > cutoff_3y]
        .groupby("CLIENT_ID")["TO_WITHOUTTAX_EUR_CONST"]
        .sum()
        .rename("SPEND_PAST_3Y")
        .reset_index()
    )
    spend_3y_6y = (
        t[(t["TRS_DATE"] > cutoff_6y) & (t["TRS_DATE"] <= cutoff_3y)]
        .groupby("CLIENT_ID")["TO_WITHOUTTAX_EUR_CONST"]
        .sum()
        .rename("SPEND_3Y_6Y")
        .reset_index()
    )

    feat_base = feat_base.merge(spend_3y,    on="CLIENT_ID", how="left")
    feat_base = feat_base.merge(spend_3y_6y, on="CLIENT_ID", how="left")
    feat_base["SPEND_PAST_3Y"] = feat_base["SPEND_PAST_3Y"].fillna(0)
    feat_base["SPEND_3Y_6Y"]   = feat_base["SPEND_3Y_6Y"].fillna(0)

    # Trend di spesa: ratio periodo recente / periodo precedente
    feat_base["SPEND_TREND"] = np.where(
        feat_base["SPEND_3Y_6Y"] > 0,
        feat_base["SPEND_PAST_3Y"] / feat_base["SPEND_3Y_6Y"],
        np.where(feat_base["SPEND_PAST_3Y"] > 0, 2.0, 1.0),
    )

    # --- N Sale e N Repair ---
    n_sales = (
        sales.groupby("CLIENT_ID").size()
        .rename("N_SALE_TRS")
        .reset_index()
    )
    n_repairs = (
        repairs.groupby("CLIENT_ID").size()
        .rename("N_REPAIR_TRS")
        .reset_index()
    )
    feat_base = feat_base.merge(n_sales,   on="CLIENT_ID", how="left")
    feat_base = feat_base.merge(n_repairs, on="CLIENT_ID", how="left")
    feat_base["N_SALE_TRS"]   = feat_base["N_SALE_TRS"].fillna(0)
    feat_base["N_REPAIR_TRS"] = feat_base["N_REPAIR_TRS"].fillna(0)
    feat_base["REPAIR_RATIO"] = feat_base["N_REPAIR_TRS"] / feat_base["N_TRANSACTIONS"]

    # --- Boutique ratio (canale) ---
    n_total    = t.groupby("CLIENT_ID").size().rename("_N_TOT")
    n_boutique = (
        t[t["CHANNEL"] == "Boutique"]
        .groupby("CLIENT_ID").size()
        .rename("_N_BTQ")
    )
    boutique_df = pd.concat([n_total, n_boutique], axis=1).fillna(0).reset_index()
    boutique_df["BOUTIQUE_RATIO"] = boutique_df["_N_BTQ"] / boutique_df["_N_TOT"]
    feat_base = feat_base.merge(
        boutique_df[["CLIENT_ID", "BOUTIQUE_RATIO"]], on="CLIENT_ID", how="left"
    )
    feat_base["BOUTIQUE_RATIO"] = feat_base["BOUTIQUE_RATIO"].fillna(0)

    # --- Stagionalità: acquisti nei mesi festivi (febbraio, dicembre) ---
    t["IS_HOLIDAY"] = t["TRS_DATE"].dt.month.isin([2, 12]).astype(int)
    holiday_ratio = (
        t.groupby("CLIENT_ID")["IS_HOLIDAY"]
        .mean()
        .rename("HOLIDAY_PURCHASE_RATIO")
        .reset_index()
    )
    feat_base = feat_base.merge(holiday_ratio, on="CLIENT_ID", how="left")

    # --- Intervallo medio tra transazioni (vettorizzato) ---
    t_sorted = t.sort_values(["CLIENT_ID", "TRS_DATE"])
    t_sorted["_PREV_DATE"] = t_sorted.groupby("CLIENT_ID")["TRS_DATE"].shift(1)
    t_sorted["_GAP_DAYS"]  = (t_sorted["TRS_DATE"] - t_sorted["_PREV_DATE"]).dt.days
    avg_gap = (
        t_sorted.groupby("CLIENT_ID")["_GAP_DAYS"]
        .mean()
        .rename("AVG_DAYS_BETWEEN_TRS")
        .reset_index()
    )
    feat_base = feat_base.merge(avg_gap, on="CLIENT_ID", how="left")
    # NaN = clienti con una sola transazione → fill 0 (nessun gap misurabile)
    # HAS_MULTIPLE_PURCHASES cattura già il caso binario
    feat_base["AVG_DAYS_BETWEEN_TRS"] = feat_base["AVG_DAYS_BETWEEN_TRS"].fillna(0)

    # --- FLAG HE (high-end) ratio dalle transazioni ---
    if "FLAG_HE" in t.columns:
        he_ratio = (
            t.groupby("CLIENT_ID")["FLAG_HE"]
            .mean()
            .rename("FLAG_HE_RATIO_TRS")
            .reset_index()
        )
        feat_base = feat_base.merge(he_ratio, on="CLIENT_ID", how="left")

    # Identificatore snapshot — aggiungi dopo tutte le feature
    feat_base["DATE_TARGET"] = date_target

    # Rimuovi colonne ausiliarie
    feat_base = feat_base.drop(
        columns=["LAST_PURCHASE_DATE", "FIRST_PURCHASE_DATE"], errors="ignore"
    )

    return feat_base


def run_transaction_features(trans, snapshots):
    """Esegue build_transaction_features per tutti gli snapshot e impila."""
    print("\n" + "=" * 60)
    print("FASE 2 — FEATURE DA TRANSACTIONS (RFM per snapshot)")
    print("=" * 60)

    results = []
    for snap in snapshots:
        snap_ts = pd.Timestamp(snap)
        print(f"  Snapshot {snap_ts.date()} ...", end=" ")
        feat = build_transaction_features(trans, snap_ts)
        print(f"{len(feat):>7} clienti, {feat.shape[1]} colonne")
        results.append(feat)

    trans_features = pd.concat(results, ignore_index=True)
    print(f"\n  Totale feature transazioni: {trans_features.shape}")

    out_path = os.path.join(FEATURES_DIR, "transaction_features.csv")
    trans_features.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path}")
    print("=== FEATURE TRANSAZIONI COMPLETATE ===")
    return trans_features


# ---------------------------------------------------------------------------
# Fase 3 — Feature da Articles (join su ARTICLE_ID)
# ---------------------------------------------------------------------------
def build_article_features(trans: pd.DataFrame,
                            articles: pd.DataFrame,
                            date_target: pd.Timestamp) -> pd.DataFrame:
    """
    Builds article-level features for a given snapshot.
    Uses LEFT JOIN to preserve transactions with unlisted articles (orfani).
    Anti-leakage: only uses transactions with TRS_DATE <= date_target.
    """
    t = trans[trans["TRS_DATE"] <= date_target].copy()
    sales_only = t[t["TRS_CATEG"] == "Sale"]

    # LEFT JOIN con Articles su ARTICLE_ID — preserva gli orfani (18.6%)
    # suffixes per evitare conflitti con FLAG_HE già presente in Transactions
    t_art = sales_only.merge(articles, on="ARTICLE_ID", how="left",
                              suffixes=("_TRS", "_ART"))

    # Seleziona le colonne corrette dopo il merge (da Articles)
    flag_he_col      = "FLAG_HE_ART"      if "FLAG_HE_ART"      in t_art.columns else "FLAG_HE"
    flag_bridal_col  = "FLAG_BRIDAL"
    flag_diamond_col = "FLAG_DIAMOND"
    wp_col           = "WORLD_PRICE"
    cat_col          = "PRODUCT_CATEGORY"

    agg_dict = {
        "AVG_ARTICLE_WORLD_PRICE": (wp_col,    "mean"),
        "MAX_ARTICLE_WORLD_PRICE": (wp_col,    "max"),
        "FLAG_HE_RATIO":           (flag_he_col,      "mean"),
        "N_DISTINCT_CATEGORIES":   (cat_col,   "nunique"),
    }
    # Aggiunge FLAG_BRIDAL e FLAG_DIAMOND solo se presenti
    if flag_bridal_col in t_art.columns:
        agg_dict["FLAG_BRIDAL_RATIO"]  = (flag_bridal_col,  "mean")
    if flag_diamond_col in t_art.columns:
        agg_dict["FLAG_DIAMOND_RATIO"] = (flag_diamond_col, "mean")

    art_features = t_art.groupby("CLIENT_ID").agg(**agg_dict).reset_index()

    art_features["DATE_TARGET"] = date_target
    return art_features


def run_article_features(trans, articles, snapshots):
    """Esegue build_article_features per tutti gli snapshot e impila."""
    print("\n" + "=" * 60)
    print("FASE 3 — FEATURE DA ARTICLES")
    print("=" * 60)

    if articles is None:
        print("  Articles.csv non disponibile — fase saltata.")
        print("  Creo DataFrame vuoto con solo CLIENT_ID e DATE_TARGET.")
        # Ritorna DataFrame vuoto strutturato per non rompere il join successivo
        empty = pd.DataFrame(columns=["CLIENT_ID", "DATE_TARGET"])
        empty.to_csv(os.path.join(FEATURES_DIR, "article_features.csv"), index=False)
        return empty

    results = []
    for snap in snapshots:
        snap_ts = pd.Timestamp(snap)
        print(f"  Snapshot {snap_ts.date()} ...", end=" ")
        feat = build_article_features(trans, articles, snap_ts)
        print(f"{len(feat):>7} clienti, {feat.shape[1]} colonne")
        results.append(feat)

    art_features = pd.concat(results, ignore_index=True)
    print(f"\n  Totale feature articoli: {art_features.shape}")

    out_path = os.path.join(FEATURES_DIR, "article_features.csv")
    art_features.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path}")
    print("=== FEATURE ARTICOLI COMPLETATE ===")
    return art_features


# ---------------------------------------------------------------------------
# Fase 4 — Feature da Aggregated_Data_clean
# ---------------------------------------------------------------------------

# Colonne con near-zero variance (da CLAUDE.md FE Step 1)
NEAR_ZERO_VAR_COLS = [
    "MAX_PRICE_IN_BTQ",   # 100% zero
    "NB_TRS_BTQ",         # 100% zero
    "TO_OTHER_HE",        # 99.99% zero
    "TO_CRC",             # >99% zero
    "TO_WEB",             # >99% zero
    "TO_MORE_10K",        # >99% zero
]

# Colonne escluse per altri motivi
ALWAYS_EXCLUDE = [
    # Target — mai usare come feature
    "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
    # AGE: troppo sparse sugli snapshot storici, usare solo AGE_KNOWN
    "AGE",
    # Missing strutturale già trasformato in flag (>84% missing per clienti single-purchase)
    "STDDEV_TIMELAPSE_TRS",
    "AVG_TIMELAPSE_PER_TRS",
    "MIN_TIMELAPSE_TRS",
    "STDDEV_PRICE",
    "TO_STDDEV_SPREAD",   # 84.6% missing strutturale — stessa policy di STDDEV_PRICE
]


def select_aggregated_features(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Selects pre-computed features from Aggregated_Data_clean.
    Esclude target, colonne ALL_* (stringhe complesse), near-zero variance,
    e colonne con missing strutturale già trasformate in flag.
    """
    # Colonne ALL_* — stringhe raw comma-separated, richiedono parsing separato
    all_cols = [c for c in agg.columns if c.startswith("ALL_")]

    exclude = set(ALWAYS_EXCLUDE + NEAR_ZERO_VAR_COLS + all_cols)
    include = [c for c in agg.columns if c not in exclude]

    selected = agg[include].copy()

    print("\n" + "=" * 60)
    print("FASE 4 — FEATURE DA AGGREGATED_DATA_CLEAN")
    print("=" * 60)
    print(f"  Colonne incluse:  {len(include)}")
    print(f"  Colonne escluse:  {len(exclude)}")
    print(f"  Dettaglio escluse:")
    print(f"    Target:              {len([c for c in ALWAYS_EXCLUDE if 'TARGET' in c])}")
    print(f"    Near-zero variance:  {len(NEAR_ZERO_VAR_COLS)}")
    print(f"    ALL_* (stringhe):    {len(all_cols)}")
    print(f"    Altre strutturali:   {len([c for c in ALWAYS_EXCLUDE if 'TARGET' not in c])}")
    print(f"  Shape selezionato: {selected.shape}")

    out_path = os.path.join(FEATURES_DIR, "aggregated_features.csv")
    selected.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path}")
    print("=== FEATURE AGGREGATED SELEZIONATE ===")
    return selected


# ---------------------------------------------------------------------------
# Fase 5 — Join e costruzione master feature set
# ---------------------------------------------------------------------------
def build_master_feature_set(agg_features: pd.DataFrame,
                              trans_features: pd.DataFrame,
                              art_features: pd.DataFrame,
                              supp: pd.DataFrame) -> pd.DataFrame:
    """
    Joins all feature sources into a single master dataset.
    One row per (CLIENT_ID, DATE_TARGET).

    Join strategy:
    - Aggregated_Data è la base (panel completo)
    - Transaction/Article features: LEFT JOIN su (CLIENT_ID, DATE_TARGET)
    - Supplementary features: LEFT JOIN solo su CLIENT_ID
      (aggregato storico — usare solo HAS_CRC e HAS_CCP come flag binarie)
    """
    print("\n" + "=" * 60)
    print("FASE 5 — JOIN MASTER FEATURE SET")
    print("=" * 60)

    master = agg_features.copy()
    n_base = len(master)
    print(f"  Base (Aggregated): {master.shape}")

    # Join transaction features
    if len(trans_features) > 0 and "DATE_TARGET" in trans_features.columns:
        trans_features["DATE_TARGET"] = pd.to_datetime(trans_features["DATE_TARGET"])
        master["DATE_TARGET"] = pd.to_datetime(master["DATE_TARGET"])
        master = master.merge(
            trans_features,
            on=["CLIENT_ID", "DATE_TARGET"],
            how="left",
            suffixes=("", "_TRS"),
        )
        print(f"  Dopo join Transactions: {master.shape}")

    # Join article features
    if len(art_features) > 0 and "DATE_TARGET" in art_features.columns:
        art_features["DATE_TARGET"] = pd.to_datetime(art_features["DATE_TARGET"])
        master = master.merge(
            art_features,
            on=["CLIENT_ID", "DATE_TARGET"],
            how="left",
            suffixes=("", "_ART"),
        )
        print(f"  Dopo join Articles:     {master.shape}")

    # Join supplementary features — solo flag binarie sicure (senza leakage)
    # N_CRC_INTERACTIONS e AVG_APPOINTMENT_DURATION esclusi (rischio leakage)
    supp_safe_cols = ["CLIENT_ID", "HAS_CRC_INTERACTION", "HAS_CCP",
                      "HAS_GIFT_REGISTERED", "HAS_CLIENTELING"]
    supp_safe = supp[[c for c in supp_safe_cols if c in supp.columns]].copy()
    master = master.merge(supp_safe, on="CLIENT_ID", how="left", suffixes=("", "_SUPP"))
    print(f"  Dopo join Supplementary: {master.shape}")

    assert len(master) == n_base, (
        f"ERRORE: il join ha cambiato il numero di righe! "
        f"{n_base} -> {len(master)}"
    )

    out_path = os.path.join(FEATURES_DIR, "master_features_all_snapshots.csv")
    master.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path}")
    return master


# ---------------------------------------------------------------------------
# Fase 6 — Log-transformation dei target
# ---------------------------------------------------------------------------
def add_log_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log-transformed targets for the Two-Part Model.
    - LOG_TARGET_*:    log1p(TARGET) per la regressione (Parte 2)
    - BINARY_TARGET_*: (TARGET > 0) per il classificatore (Parte 1)
    log1p gestisce i valori zero: log1p(0) = 0.
    """
    for col in ["TARGET_3Y", "TARGET_5Y"]:
        if col in df.columns:
            df[f"LOG_{col}"]    = np.log1p(df[col])
            df[f"BINARY_{col}"] = (df[col] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Fase 7 — Split train/test
# ---------------------------------------------------------------------------
def split_train_test(df: pd.DataFrame):
    """
    Temporal holdout split:
    - Train: snapshot 2006-2018 (5 snapshot)
    - Test:  snapshot 2021     (1 snapshot, completamente isolato)
    """
    print("\n" + "=" * 60)
    print("FASE 7 — SPLIT TRAIN / TEST")
    print("=" * 60)

    df["DATE_TARGET"] = pd.to_datetime(df["DATE_TARGET"])
    train = df[df["DATE_TARGET"] < "2021-01-01"].copy()
    test  = df[df["DATE_TARGET"] == "2021-01-01"].copy()

    overlap = train["CLIENT_ID"].isin(test["CLIENT_ID"]).sum()

    print(f"  Train: {train.shape}  — snapshot 2006-2018")
    print(f"  Test:  {test.shape}   — snapshot 2021")
    print(f"  CLIENT_ID in comune train/test: {overlap:,}")
    print(f"  (Atteso: ~412.571 — stesso pool clienti su snapshot diversi)")

    train.to_csv(os.path.join(FEATURES_DIR, "train_features.csv"), index=False)
    test.to_csv(os.path.join(FEATURES_DIR,  "test_features.csv"),  index=False)
    print(f"  Salvato: data/features/train_features.csv")
    print(f"  Salvato: data/features/test_features.csv")
    return train, test


# ---------------------------------------------------------------------------
# Fase 8 — Validazione feature set
# ---------------------------------------------------------------------------
def validate_feature_set(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """Validates the final feature set before modeling."""
    print("\n" + "=" * 60)
    print("FASE 8 — VALIDAZIONE FEATURE SET")
    print("=" * 60)

    checks = []

    # 1. Test set non vuoto
    checks.append(("Test set non vuoto", len(test) > 0))

    # 2. Nessuna colonna con varianza zero nel train (escludo target e flag)
    skip_cols = {
        "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
        "LOG_TARGET_3Y", "LOG_TARGET_5Y",
        "BINARY_TARGET_3Y", "BINARY_TARGET_5Y",
    }
    num_cols   = train.select_dtypes("number").columns
    check_cols = [c for c in num_cols if c not in skip_cols]
    zero_var   = [c for c in check_cols if train[c].std() == 0]
    checks.append(("Zero variance features assenti", len(zero_var) == 0))
    if zero_var:
        print(f"  ATTENZIONE colonne zero-variance: {zero_var}")

    # 3. Colonne target binarie presenti e corrette
    checks.append(("BINARY_TARGET_3Y presente",
                   "BINARY_TARGET_3Y" in train.columns))
    if "BINARY_TARGET_3Y" in train.columns:
        checks.append(("BINARY_TARGET_3Y è 0/1",
                       train["BINARY_TARGET_3Y"].isin([0, 1]).all()))
        pct_positive = train["BINARY_TARGET_3Y"].mean()
        print(f"  BINARY_TARGET_3Y positivi (train): {pct_positive:.1%}")

    # 4. LOG_TARGET_3Y non negativo
    if "LOG_TARGET_3Y" in train.columns:
        checks.append(("LOG_TARGET_3Y >= 0",
                       (train["LOG_TARGET_3Y"] >= 0).all()))

    # 5. TARGET originali non modificati
    if "TARGET_3Y" in train.columns:
        checks.append(("TARGET_3Y non negativo",
                       (train["TARGET_3Y"] >= 0).all()))

    # 6. Colonne con >50% missing nel train
    missing_pct  = train.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5].sort_values(ascending=False)
    checks.append((f"Nessuna colonna con >50% missing nel train",
                   len(high_missing) == 0))
    if len(high_missing) > 0:
        print(f"  Colonne con >50% missing ({len(high_missing)}):")
        for col, pct in high_missing.head(10).items():
            print(f"    {col}: {pct:.1%}")

    # 7. Nessun CLIENT_ID nullo
    checks.append(("CLIENT_ID non nullo nel train",
                   train["CLIENT_ID"].notna().all()))
    checks.append(("CLIENT_ID non nullo nel test",
                   test["CLIENT_ID"].notna().all()))

    # Report
    print()
    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("Risultato: TUTTI PASS")
    else:
        print("Risultato: ALCUNI FAIL — verificare output sopra")

    return all_pass


# ---------------------------------------------------------------------------
# Fase 9 — Report feature engineering
# ---------------------------------------------------------------------------
def generate_report(train: pd.DataFrame, test: pd.DataFrame,
                    agg_features: pd.DataFrame):
    """
    Genera report del feature set e calcola correlazioni con TARGET_3Y.
    Salva:
      output/tables/feature_correlations.csv
      output/tables/feature_engineering_report.csv
    """
    print("\n" + "=" * 60)
    print("FASE 9 — REPORT FEATURE ENGINEERING")
    print("=" * 60)

    # --- Categorizzazione feature ---
    def categorize(col):
        rfm_cols = {
            "N_TRANSACTIONS", "TOTAL_SPEND", "AVG_SPEND_PER_TRS",
            "MAX_SINGLE_SPEND", "N_DISTINCT_ARTICLES", "RECENCY_DAYS",
            "TENURE_DAYS", "SPEND_PAST_3Y", "SPEND_3Y_6Y", "SPEND_TREND",
            "N_SALE_TRS", "N_REPAIR_TRS", "REPAIR_RATIO", "BOUTIQUE_RATIO",
            "HOLIDAY_PURCHASE_RATIO", "AVG_DAYS_BETWEEN_TRS", "FLAG_HE_RATIO_TRS",
        }
        art_cols = {
            "AVG_ARTICLE_WORLD_PRICE", "MAX_ARTICLE_WORLD_PRICE",
            "FLAG_HE_RATIO", "FLAG_BRIDAL_RATIO", "FLAG_DIAMOND_RATIO",
            "N_DISTINCT_CATEGORIES",
        }
        supp_cols = {
            "HAS_CRC_INTERACTION", "HAS_CCP", "HAS_GIFT_REGISTERED", "HAS_CLIENTELING",
        }
        if col in rfm_cols:
            return "RFM_Transactions"
        if col in art_cols:
            return "Articles"
        if col in supp_cols:
            return "Supplementary"
        if col in {"TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
                   "LOG_TARGET_3Y", "LOG_TARGET_5Y",
                   "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"}:
            return "Target"
        if col in {"CLIENT_ID", "DATE_TARGET"}:
            return "Identifier"
        return "Aggregated"

    num_cols = train.select_dtypes("number").columns.tolist()
    all_cols = train.columns.tolist()

    # Statistiche generali
    n_feat_total = len([c for c in all_cols
                        if categorize(c) not in ("Target", "Identifier")])
    cat_counts   = {}
    for c in all_cols:
        cat = categorize(c)
        if cat not in ("Target", "Identifier"):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"  Feature totali nel train (esclusi target/id): {n_feat_total}")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    missing_mean = train[num_cols].isnull().mean()
    print(f"\n  % missing media nel train (numeriche): {missing_mean.mean():.1%}")
    top10_missing = missing_mean.sort_values(ascending=False).head(10)
    print(f"  Top 10 colonne per % missing:")
    for col, pct in top10_missing.items():
        print(f"    {col}: {pct:.1%}")

    # Distribuzione target nel train
    if "BINARY_TARGET_3Y" in train.columns:
        pct_pos = train["BINARY_TARGET_3Y"].mean()
        print(f"\n  BINARY_TARGET_3Y = 1 (spenderà): {pct_pos:.1%}")
    if "LOG_TARGET_3Y" in train.columns:
        positives = train.loc[train["LOG_TARGET_3Y"] > 0, "LOG_TARGET_3Y"]
        print(f"  LOG_TARGET_3Y sui positivi: "
              f"media={positives.mean():.2f}, "
              f"std={positives.std():.2f}, "
              f"p95={positives.quantile(0.95):.2f}")

    # Correlazioni con TARGET_3Y
    corr_report = pd.DataFrame()
    if "TARGET_3Y" in train.columns:
        corr = (
            train[num_cols]
            .corr()["TARGET_3Y"]
            .abs()
            .sort_values(ascending=False)
            .reset_index()
        )
        corr.columns = ["feature", "abs_corr_with_TARGET_3Y"]
        corr["categoria"] = corr["feature"].apply(categorize)
        corr = corr[corr["feature"] != "TARGET_3Y"]

        print(f"\n  Top 20 feature per correlazione |r| con TARGET_3Y:")
        for _, row in corr.head(20).iterrows():
            print(f"    {row['feature']:40s}  {row['abs_corr_with_TARGET_3Y']:.4f}")

        corr_path = os.path.join(TABLES_DIR, "feature_correlations.csv")
        corr.to_csv(corr_path, index=False)
        print(f"\n  Salvato: {corr_path}")
        corr_report = corr

    # Feature escluse
    all_agg_cols = set(agg_features.columns)
    excluded_info = []
    for col in agg_features.columns:
        if col.startswith("ALL_"):
            excluded_info.append({"colonna": col, "motivo": "ALL_* — stringa CSV non parsata"})
    for col in ALWAYS_EXCLUDE:
        if col in all_agg_cols:
            excluded_info.append({"colonna": col, "motivo": "Esclusa per policy (CLAUDE.md)"})
    for col in NEAR_ZERO_VAR_COLS:
        if col in all_agg_cols:
            excluded_info.append({"colonna": col, "motivo": "Near-zero variance"})

    excl_df = pd.DataFrame(excluded_info)

    # Report complessivo
    report_rows = []
    report_rows.append({"sezione": "Overview", "metrica": "Feature totali train",
                        "valore": str(n_feat_total)})
    report_rows.append({"sezione": "Overview", "metrica": "Righe train",
                        "valore": str(len(train))})
    report_rows.append({"sezione": "Overview", "metrica": "Righe test",
                        "valore": str(len(test))})
    report_rows.append({"sezione": "Overview", "metrica": "Colonne totali train",
                        "valore": str(train.shape[1])})
    report_rows.append({"sezione": "Missing", "metrica": "% missing media (numeriche)",
                        "valore": f"{missing_mean.mean():.1%}"})
    for col, pct in top10_missing.items():
        report_rows.append({"sezione": "Missing_top10", "metrica": col,
                            "valore": f"{pct:.1%}"})
    for cat, cnt in sorted(cat_counts.items()):
        report_rows.append({"sezione": "Feature_per_categoria", "metrica": cat,
                            "valore": str(cnt)})
    if "BINARY_TARGET_3Y" in train.columns:
        report_rows.append({"sezione": "Target", "metrica": "% BINARY_TARGET_3Y=1",
                            "valore": f"{train['BINARY_TARGET_3Y'].mean():.1%}"})

    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(TABLES_DIR, "feature_engineering_report.csv")
    report_df.to_csv(report_path, index=False)

    excl_path = os.path.join(TABLES_DIR, "feature_engineering_excluded.csv")
    excl_df.to_csv(excl_path, index=False)

    print(f"\n  Salvato: {report_path}")
    print(f"  Salvato: {excl_path} ({len(excl_df)} colonne escluse documentate)")
    print("=== FEATURE ENGINEERING REPORT ===")

    return corr_report


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_all():
    print("\n" + "=" * 60)
    print(" CARTIER QTEM — FEATURE ENGINEERING PIPELINE ")
    print("=" * 60)

    # Fase 1 — Caricamento
    agg, trans, articles, supp, snapshots, train_snaps, test_snap = load_data()

    # Fase 2 — Transaction features (RFM)
    trans_features = run_transaction_features(trans, snapshots)

    # Fase 3 — Article features
    art_features = run_article_features(trans, articles, snapshots)

    # Fase 4 — Aggregated features
    agg_features = select_aggregated_features(agg)

    # Fase 5 — Master feature set
    master = build_master_feature_set(agg_features, trans_features,
                                      art_features, supp)

    # Fase 6 — Log-transform target
    print("\n" + "=" * 60)
    print("FASE 6 — LOG-TRANSFORM TARGET")
    print("=" * 60)

    # Reintegra TARGET nel master (erano in agg ma esclusi da agg_features)
    target_cols = ["CLIENT_ID", "DATE_TARGET", "TARGET_3Y", "TARGET_5Y", "TARGET_10Y"]
    targets = agg[[c for c in target_cols if c in agg.columns]].copy()
    targets["DATE_TARGET"] = pd.to_datetime(targets["DATE_TARGET"])
    master["DATE_TARGET"]  = pd.to_datetime(master["DATE_TARGET"])

    master_with_targets = master.merge(targets, on=["CLIENT_ID", "DATE_TARGET"],
                                       how="left", suffixes=("", "_orig"))
    master_with_targets = add_log_targets(master_with_targets)

    for col in ["TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
                "LOG_TARGET_3Y", "LOG_TARGET_5Y",
                "BINARY_TARGET_3Y", "BINARY_TARGET_5Y"]:
        if col in master_with_targets.columns:
            print(f"  {col}: non-null={master_with_targets[col].notna().sum():,}, "
                  f"min={master_with_targets[col].min():.4f}, "
                  f"max={master_with_targets[col].max():.2f}")

    out_path = os.path.join(FEATURES_DIR, "master_features_with_targets.csv")
    master_with_targets.to_csv(out_path, index=False)
    print(f"  Salvato: {out_path}")

    # Fase 7 — Train / Test split
    train, test = split_train_test(master_with_targets)

    # Fase 8 — Validazione
    all_pass = validate_feature_set(train, test)

    # Fase 9 — Report
    generate_report(train, test, agg_features)

    # Riepilogo finale
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETATA")
    print("=" * 60)
    print(f"  Train set: {train.shape}")
    print(f"  Test set:  {test.shape}")
    print(f"  Validazione: {'OK — tutti i check passati' if all_pass else 'ATTENZIONE — verificare FAIL'}")
    print()
    print("  Output generati:")
    for fname in [
        "transaction_features.csv",
        "article_features.csv",
        "aggregated_features.csv",
        "master_features_all_snapshots.csv",
        "master_features_with_targets.csv",
        "train_features.csv",
        "test_features.csv",
    ]:
        fpath = os.path.join(FEATURES_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    data/features/{fname}  ({size_mb:.1f} MB)")
        else:
            print(f"    data/features/{fname}  [MANCANTE]")

    return train, test


if __name__ == "__main__":
    train, test = run_all()
