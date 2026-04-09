"""
Feature Selection — Cartier QTEM Data Challenge
================================================
Rimuove ridondanze e multicollinearità dal feature set prodotto da feature_engineering.py.

Output:
  data/features/train_features_final.csv
  data/features/test_features_final.csv
  output/tables/high_correlation_pairs.csv
  output/tables/feature_selection_report.csv

Eseguire con:
  python scripts/feature_selection.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
TABLES_DIR   = os.path.join(BASE_DIR, "output", "tables")

# ---------------------------------------------------------------------------
# Colonne mai rimuovibili
# ---------------------------------------------------------------------------
NEVER_REMOVE = {
    "TARGET_3Y", "TARGET_5Y", "TARGET_10Y",
    "LOG_TARGET_3Y", "LOG_TARGET_5Y",
    "BINARY_TARGET_3Y", "BINARY_TARGET_5Y",
    "CLIENT_ID", "DATE_TARGET",
}

# Duplicati RFM rispetto ad Aggregated
RFM_DUPLICATES = [
    "SPEND_PAST_3Y",      # ≈ TO_PAST_3Y
    "TOTAL_SPEND",        # ≈ TO_FULL_HIST
    "N_TRANSACTIONS",     # ≈ NB_TRS_FULL_HIST
    "AVG_SPEND_PER_TRS",  # derivabile da TO_FULL_HIST / NB_TRS
    "N_SALE_TRS",         # ≈ NB_TRS_FULL_HIST (solo Sale)
    "N_REPAIR_TRS",       # ≈ HAS_REPAIR_HISTORY
    "TENURE_DAYS",        # ≈ SENIORITY
]

# Feature near-zero variance già identificate in FE Step 1
KNOWN_NEAR_ZERO = [
    "MAX_PRICE_IN_BTQ",
    "NB_TRS_BTQ",
    "TO_OTHER_HE",
]

# Feature RFM uniche da mantenere (nessun equivalente in Aggregated)
# FLAG_HE_RATIO_TRS: rimosso — >99% zero nel train
# RECENCY_DAYS: rimosso — r=0.955 con RECENCY (Aggregated), quasi identica
# MAX_SINGLE_SPEND: rimosso — r=1.0 con MAX_PRICE_PER_PDT (Aggregated)
RFM_KEEP = {
    "BOUTIQUE_RATIO",
    "HOLIDAY_PURCHASE_RATIO",
    "AVG_DAYS_BETWEEN_TRS",
    "SPEND_TREND",
    "REPAIR_RATIO",
    "N_DISTINCT_ARTICLES",
    "SPEND_3Y_6Y",
}


def load_data():
    print("=" * 60)
    print("STEP 1 — CARICAMENTO")
    print("=" * 60)
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_features.csv"), low_memory=False)
    test  = pd.read_csv(os.path.join(FEATURES_DIR, "test_features.csv"),  low_memory=False)
    print(f"  Train originale: {train.shape}")
    print(f"  Test originale:  {test.shape}")
    return train, test


def find_near_zero(train: pd.DataFrame) -> list:
    """Identifica colonne con >99% zeri nel train (esclusi target e id)."""
    print("\n" + "=" * 60)
    print("STEP 3 — NEAR-ZERO VARIANCE")
    print("=" * 60)
    num_cols = train.select_dtypes(include=[np.number]).columns
    pct_zero = (train[num_cols] == 0).mean()
    near_zero = [
        c for c in pct_zero[pct_zero > 0.99].index
        if c not in NEVER_REMOVE
    ]
    print(f"  Colonne con >99% zero: {near_zero}")
    return near_zero, pct_zero


def find_high_corr_pairs(train: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Calcola coppie di feature con correlazione assoluta > threshold."""
    print("\n" + "=" * 60)
    print("STEP 4 — MATRICE DI CORRELAZIONE (campione 50k)")
    print("=" * 60)

    target_cols = list(NEVER_REMOVE)
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in target_cols]

    sample = train[feat_cols].sample(n=min(50_000, len(train)), random_state=42)
    corr   = sample.corr().abs()

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val > threshold:
                pairs.append({
                    "feature_1":   cols[i],
                    "feature_2":   cols[j],
                    "correlation": round(val, 4),
                })

    df = pd.DataFrame(pairs).sort_values("correlation", ascending=False)
    print(f"  Coppie con r > {threshold}: {len(df)}")
    if len(df):
        print(df.head(25).to_string(index=False))

    out = os.path.join(TABLES_DIR, "high_correlation_pairs.csv")
    df.to_csv(out, index=False)
    print(f"\n  Salvato: {out}")
    return df, corr.columns.tolist()


def build_removal_set(near_zero: list, high_corr_df: pd.DataFrame,
                      feature_cols: list) -> set:
    """Costruisce l'insieme finale di colonne da rimuovere."""
    print("\n" + "=" * 60)
    print("STEP 5 — LISTA COLONNE DA RIMUOVERE")
    print("=" * 60)

    to_remove = set(RFM_DUPLICATES + KNOWN_NEAR_ZERO + near_zero)

    # Rimuove aggiuntivi r>0.95 tra RFM e Aggregated
    rfm_all = set(RFM_DUPLICATES) | set(feature_cols) - NEVER_REMOVE
    for _, row in high_corr_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        if f1 in RFM_DUPLICATES and f2 not in NEVER_REMOVE:
            to_remove.add(f1)
        if f2 in RFM_DUPLICATES and f1 not in NEVER_REMOVE:
            to_remove.add(f2)

    # Mai rimuovere target/id e feature RFM uniche
    to_remove -= NEVER_REMOVE
    to_remove -= RFM_KEEP

    # Segnala coppie Aggregated-Aggregated ad alta correlazione (non previste)
    # — documentate ma non rimosse automaticamente
    unexpected = []
    for _, row in high_corr_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        if f1 not in to_remove and f2 not in to_remove \
                and f1 not in NEVER_REMOVE and f2 not in NEVER_REMOVE:
            unexpected.append(f"  r={row['correlation']:.4f}  {f1}  <->  {f2}")
    if unexpected:
        print(f"\n  COPPIE r>0.95 NON RIMOSSE (entrambe Aggregated — decisione manuale):")
        for u in unexpected:
            print(u)

    print(f"  Totale da rimuovere: {len(to_remove)}")
    for c in sorted(to_remove):
        print(f"    - {c}")
    return to_remove


def apply_selection(train, test, to_remove: set):
    """Applica la stessa selezione a train e test."""
    print("\n" + "=" * 60)
    print("STEP 6 — APPLICAZIONE SELEZIONE")
    print("=" * 60)

    # Rimuovi solo colonne che esistono
    actual_remove = [c for c in to_remove if c in train.columns]
    train_f = train.drop(columns=actual_remove)
    test_f  = test.drop(columns=actual_remove)

    assert list(train_f.columns) == list(test_f.columns), \
        "ERRORE CRITICO: train e test hanno colonne diverse!"

    print(f"  Feature rimosse:   {len(actual_remove)}")
    print(f"  Feature mantenute: {train_f.shape[1]}")
    print(f"  Train finale: {train_f.shape}")
    print(f"  Test finale:  {test_f.shape}")
    return train_f, test_f


def save_report(train, train_f, to_remove, near_zero, pct_zero):
    """Salva il report di selezione."""
    print("\n" + "=" * 60)
    print("STEP 7 — REPORT SELEZIONE")
    print("=" * 60)

    skip = NEVER_REMOVE
    all_features = [c for c in train.columns if c not in skip]

    rows = []
    for col in all_features:
        if col in to_remove:
            if col in RFM_DUPLICATES:
                reason = "Duplicato RFM — equivalente Aggregated già presente"
            elif col in KNOWN_NEAR_ZERO or col in near_zero:
                reason = "Zero/near-zero variance (>99% zero)"
            else:
                reason = "Alta correlazione con feature Aggregated (r>0.95)"
            status = "RIMOSSA"
        else:
            reason = "Mantenuta"
            status = "MANTENUTA"

        rows.append({
            "feature":      col,
            "status":       status,
            "motivo":       reason,
            "pct_zero":     round(float(pct_zero.get(col, 0)), 4),
            "pct_missing":  round(float(train[col].isnull().mean()), 4)
                            if col in train.columns else None,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(TABLES_DIR, "feature_selection_report.csv")
    df.to_csv(out, index=False)

    kept    = df[df["status"] == "MANTENUTA"]
    removed = df[df["status"] == "RIMOSSA"]
    print(f"  Feature mantenute: {len(kept)}")
    print(f"  Feature rimosse:   {len(removed)}")
    print(f"\n  Breakdown rimozioni:")
    print(removed["motivo"].value_counts().to_string())
    print(f"\n  Salvato: {out}")


def validate(train_f, test_f, rfm_duplicates=RFM_DUPLICATES):
    """Valida il feature set finale."""
    print("\n" + "=" * 60)
    print("STEP 9 — VALIDAZIONE")
    print("=" * 60)

    checks = []

    checks.append(("Train e test allineati",
                   list(train_f.columns) == list(test_f.columns)))

    feat_cols = [c for c in train_f.columns if c not in NEVER_REMOVE]
    pct_zero_f = (train_f[feat_cols] == 0).mean()
    still_nz = pct_zero_f[pct_zero_f > 0.99].index.tolist()
    checks.append(("Nessuna feature near-zero rimasta", len(still_nz) == 0))
    if still_nz:
        print(f"  ATTENZIONE still near-zero: {still_nz}")

    for t in ["TARGET_3Y", "BINARY_TARGET_3Y", "LOG_TARGET_3Y"]:
        checks.append((f"{t} presente", t in train_f.columns))

    rfm_residui = [c for c in rfm_duplicates if c in train_f.columns]
    checks.append(("Nessun duplicato RFM rimasto", len(rfm_residui) == 0))
    if rfm_residui:
        print(f"  ATTENZIONE RFM residui: {rfm_residui}")

    checks.append(("CLIENT_ID presente",   "CLIENT_ID"   in train_f.columns))
    checks.append(("DATE_TARGET presente", "DATE_TARGET" in train_f.columns))

    print()
    all_pass = True
    for name, result in checks:
        if not result:
            all_pass = False
        print(f"  [{'PASS' if result else 'FAIL'}] {name}")

    print(f"\nRisultato: {'TUTTI PASS' if all_pass else 'ALCUNI FAIL'}")
    return all_pass


def print_final_columns(df):
    feat_cols = [c for c in df.columns if c not in NEVER_REMOVE]
    print(f"\n  Feature finali ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"    {c}")


def run_all():
    print("\n" + "=" * 60)
    print(" CARTIER QTEM — FEATURE SELECTION PIPELINE ")
    print("=" * 60)

    train, test = load_data()

    near_zero, pct_zero = find_near_zero(train)

    high_corr_df, feature_cols = find_high_corr_pairs(train, threshold=0.95)

    to_remove = build_removal_set(near_zero, high_corr_df, feature_cols)

    train_f, test_f = apply_selection(train, test, to_remove)

    save_report(train, train_f, to_remove, near_zero, pct_zero)

    # Salva
    print("\n" + "=" * 60)
    print("STEP 8 — SALVATAGGIO")
    print("=" * 60)
    train_f.to_csv(os.path.join(FEATURES_DIR, "train_features_final.csv"), index=False)
    test_f.to_csv( os.path.join(FEATURES_DIR, "test_features_final.csv"),  index=False)
    print(f"  Salvato: data/features/train_features_final.csv  "
          f"({os.path.getsize(os.path.join(FEATURES_DIR, 'train_features_final.csv'))/1024/1024:.1f} MB)")
    print(f"  Salvato: data/features/test_features_final.csv   "
          f"({os.path.getsize(os.path.join(FEATURES_DIR, 'test_features_final.csv'))/1024/1024:.1f} MB)")

    validate(train_f, test_f)

    print_final_columns(train_f)

    print("\n" + "=" * 60)
    print("=== FEATURE SET FINALE ===")
    print(f"  Train: {train_f.shape}  — pronto per modeling")
    print(f"  Test:  {test_f.shape}   — isolato, non toccare fino a valutazione")
    print("=" * 60)

    return train_f, test_f


if __name__ == "__main__":
    run_all()
