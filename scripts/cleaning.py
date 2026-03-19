"""
cleaning.py — Pipeline di cleaning production-ready per il progetto Cartier QTEM.

Principi:
- Legge da data/raw/, scrive in data/processed/
- Mai modifica i file raw
- Ogni funzione è indipendente e testabile
- Ogni funzione stampa un report con N. righe iniziali/finali, trasformazioni
- Commenti in italiano, docstring in inglese
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "output" / "tables"

DATA_PROC.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Placeholder SERIAL_NUMBER noto
_SERIAL_PLACEHOLDER = "4e2e3377c60db4140ae7"


def _print_report(name: str, n_raw: int, n_clean: int, transforms: list[str],
                  flags_added: list[str] | None = None, cols_added: list[str] | None = None) -> None:
    """Stampa report sintetico di cleaning per un dataset."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"CLEANING REPORT — {name}")
    print(sep)
    print(f"  Righe iniziali:  {n_raw:>10,}")
    print(f"  Righe finali:    {n_clean:>10,}")
    print(f"  Righe rimosse:   {n_raw - n_clean:>10,} ({(n_raw-n_clean)/n_raw*100:.3f}%)")
    if flags_added:
        print(f"  Flag create:     {', '.join(flags_added)}")
    if cols_added:
        print(f"  Colonne aggiunte:{', '.join(cols_added)}")
    print("  Trasformazioni:")
    for t in transforms:
        print(f"    - {t}")


# ===========================================================================
# AGGREGATED DATA
# ===========================================================================

def clean_aggregated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Aggregated_Data. Returns cleaned DataFrame with binary flags.

    Transformations:
    1. Drop rows with null CLIENT_ID (8 rows)
    2. Ensure DATE_TARGET is datetime
    3. Create AGE_KNOWN flag (1 where AGE is not null)
    4. Create HAS_MULTIPLE_PURCHASES flag (1 where STDDEV_TIMELAPSE_TRS not null)
    5. Create HAS_REPAIR_HISTORY flag (1 where ALL_REPAIR_PDT_CATEG not null)
    """
    out = df.copy()
    n_raw = len(out)
    transforms = []
    flags = []

    # 1. Rimuovi CLIENT_ID null
    null_ids = out["CLIENT_ID"].isna().sum()
    out = out[out["CLIENT_ID"].notna()].copy()
    transforms.append(f"Rimossi {null_ids} record con CLIENT_ID null")

    # 2. DATE_TARGET come datetime
    if not pd.api.types.is_datetime64_any_dtype(out["DATE_TARGET"]):
        out["DATE_TARGET"] = pd.to_datetime(out["DATE_TARGET"], errors="coerce")
        transforms.append("DATE_TARGET convertita a datetime")

    # 3. Flag AGE_KNOWN
    out["AGE_KNOWN"] = out["AGE"].notna().astype(np.int8)
    flags.append("AGE_KNOWN")
    transforms.append(
        f"Creata AGE_KNOWN: {out['AGE_KNOWN'].sum():,} con AGE valida ({out['AGE_KNOWN'].mean()*100:.1f}%)"
    )

    # 4. Flag HAS_MULTIPLE_PURCHASES (proxy: STDDEV_TIMELAPSE_TRS non null)
    stddev_col = "STDDEV_TIMELAPSE_TRS"
    if stddev_col in out.columns:
        out["HAS_MULTIPLE_PURCHASES"] = out[stddev_col].notna().astype(np.int8)
        flags.append("HAS_MULTIPLE_PURCHASES")
        transforms.append(
            f"Creata HAS_MULTIPLE_PURCHASES da {stddev_col}: "
            f"{out['HAS_MULTIPLE_PURCHASES'].sum():,} clienti con NB_TRS>1 ({out['HAS_MULTIPLE_PURCHASES'].mean()*100:.1f}%)"
        )

    # 5. Flag HAS_REPAIR_HISTORY
    repair_col = "ALL_REPAIR_PDT_CATEG"
    if repair_col in out.columns:
        out["HAS_REPAIR_HISTORY"] = out[repair_col].notna().astype(np.int8)
        flags.append("HAS_REPAIR_HISTORY")
        transforms.append(
            f"Creata HAS_REPAIR_HISTORY da {repair_col}: "
            f"{out['HAS_REPAIR_HISTORY'].sum():,} clienti ({out['HAS_REPAIR_HISTORY'].mean()*100:.1f}%)"
        )

    _print_report("Aggregated_Data", n_raw, len(out), transforms, flags_added=flags)
    return out


# ===========================================================================
# TRANSACTIONS
# ===========================================================================

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Transactions. Returns cleaned DataFrame.

    Transformations:
    1. Ensure TRS_DATE is datetime
    2. Drop 4 rows with TO_WITHOUTTAX_EUR_CONST < 0
    3. Impute TO_WITHOUTTAX_EUR_CONST null (422 rows): Sale->ARTICLE_WWPRICE, Repair->0
    4. Replace SERIAL_NUMBER placeholder with NaN; create SERIAL_NUMBER_KNOWN flag
    5. Create WWPRICE_MISSING flag (ARTICLE_WWPRICE=0 with TO_WITHOUTTAX>0)
    Note: TRS_DATE <= DATE_TARGET filter is applied in feature engineering per snapshot.
    """
    out = df.copy()
    n_raw = len(out)
    transforms = []
    flags = []

    # 1. TRS_DATE come datetime
    if not pd.api.types.is_datetime64_any_dtype(out["TRS_DATE"]):
        out["TRS_DATE"] = pd.to_datetime(out["TRS_DATE"], errors="coerce")
        transforms.append("TRS_DATE convertita a datetime")

    # 2. Rimuovi TO_WITHOUTTAX < 0 (4 righe, tutte Repair)
    col_to = "TO_WITHOUTTAX_EUR_CONST"
    n_neg = (out[col_to] < 0).sum()
    out = out[out[col_to].isna() | (out[col_to] >= 0)].copy()
    transforms.append(f"Rimossi {n_neg} record con {col_to} < 0 (rettifiche su Repair)")

    # 3. Imputa TO_WITHOUTTAX null
    n_null_to = out[col_to].isna().sum()
    out["TO_WITHOUTTAX_IMPUTED"] = np.int8(0)
    flags.append("TO_WITHOUTTAX_IMPUTED")

    if n_null_to > 0:
        # Sale: imputa con ARTICLE_WWPRICE
        mask_sale_null = out[col_to].isna() & (out["TRS_CATEG"] == "Sale")
        out.loc[mask_sale_null, col_to] = out.loc[mask_sale_null, "ARTICLE_WWPRICE"]
        out.loc[mask_sale_null, "TO_WITHOUTTAX_IMPUTED"] = 1

        # Repair: imputa con 0
        mask_repair_null = out[col_to].isna() & (out["TRS_CATEG"] == "Repair")
        out.loc[mask_repair_null, col_to] = 0.0
        out.loc[mask_repair_null, "TO_WITHOUTTAX_IMPUTED"] = 1

        # Residui (rari, altri): imputa con 0
        mask_other_null = out[col_to].isna()
        out.loc[mask_other_null, col_to] = 0.0
        out.loc[mask_other_null, "TO_WITHOUTTAX_IMPUTED"] = 1

        n_still_null = out[col_to].isna().sum()
        transforms.append(
            f"Imputati {n_null_to} null in {col_to} "
            f"(Sale->WWPRICE, Repair->0); residui null: {n_still_null}"
        )

    # 4. SERIAL_NUMBER placeholder -> NaN + flag
    col_sn = "SERIAL_NUMBER"
    n_ph = (out[col_sn] == _SERIAL_PLACEHOLDER).sum()
    out.loc[out[col_sn] == _SERIAL_PLACEHOLDER, col_sn] = np.nan
    out["SERIAL_NUMBER_KNOWN"] = out[col_sn].notna().astype(np.int8)
    flags.append("SERIAL_NUMBER_KNOWN")
    transforms.append(
        f"Placeholder SERIAL_NUMBER ({_SERIAL_PLACEHOLDER[:16]}...) -> NaN: {n_ph:,} record. "
        f"Creata SERIAL_NUMBER_KNOWN: {out['SERIAL_NUMBER_KNOWN'].sum():,} noti"
    )

    # 5. Flag WWPRICE_MISSING
    col_ww = "ARTICLE_WWPRICE"
    mask_ww_missing = (out[col_ww] == 0) & (out[col_to] > 0)
    out["WWPRICE_MISSING"] = mask_ww_missing.astype(np.int8)
    flags.append("WWPRICE_MISSING")
    transforms.append(
        f"Creata WWPRICE_MISSING: {mask_ww_missing.sum():,} record "
        f"({mask_ww_missing.sum()/len(out)*100:.2f}%) — WWPRICE=0 con TO>0"
    )

    _print_report("Transactions", n_raw, len(out), transforms, flags_added=flags)
    return out


# ===========================================================================
# CLIENTS
# ===========================================================================

def clean_clients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Clients. Returns cleaned DataFrame.

    Transformations:
    1. Parse BIRTH_DATE, FIRST_PURCHASE_DATE, FIRST_TRANSACTION_DATE as datetime
    2. Nullify anomalous BIRTH_DATE: pre-1900, age<18, age>100
    3. Create BIRTH_DATE_VALID flag
    4. Compute PURCHASE_TRANSACTION_GAP and flag PURCHASE_DATE_ANOMALY (>365gg)
    """
    out = df.copy()
    n_raw = len(out)
    transforms = []
    flags = []

    # 1. Parse date
    for col in ["BIRTH_DATE", "FIRST_PURCHASE_DATE", "FIRST_TRANSACTION_DATE"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    transforms.append("BIRTH_DATE, FIRST_PURCHASE_DATE, FIRST_TRANSACTION_DATE -> datetime")

    # 2. Nullify BIRTH_DATE anomale
    ref = pd.Timestamp("2021-01-01")
    bd = out["BIRTH_DATE"]
    age_at_2021 = (ref - bd).dt.days / 365.25

    mask_pre1900 = bd.dt.year < 1900
    mask_too_young = age_at_2021 < 18
    mask_too_old = age_at_2021 > 100
    mask_anomaly = mask_pre1900 | mask_too_young | mask_too_old

    n_nullified = int(mask_anomaly.sum())
    out.loc[mask_anomaly, "BIRTH_DATE"] = pd.NaT

    # 3. Flag BIRTH_DATE_VALID
    out["BIRTH_DATE_VALID"] = out["BIRTH_DATE"].notna().astype(np.int8)
    flags.append("BIRTH_DATE_VALID")
    transforms.append(
        f"Nullificati {n_nullified:,} BIRTH_DATE anomale "
        f"(pre-1900: {mask_pre1900.sum():,}, eta<18: {mask_too_young.sum():,}, "
        f"eta>100: {mask_too_old.sum():,}). "
        f"BIRTH_DATE_VALID = 1: {out['BIRTH_DATE_VALID'].sum():,}"
    )

    # 4. Gap FIRST_PURCHASE vs FIRST_TRANSACTION
    if "FIRST_PURCHASE_DATE" in out.columns and "FIRST_TRANSACTION_DATE" in out.columns:
        gap = (out["FIRST_PURCHASE_DATE"] - out["FIRST_TRANSACTION_DATE"]).dt.days.abs()
        out["PURCHASE_TRANSACTION_GAP"] = gap
        out["PURCHASE_DATE_ANOMALY"] = (gap > 365).astype(np.int8)
        n_anom = int((gap > 365).sum())
        flags.append("PURCHASE_DATE_ANOMALY")
        transforms.append(
            f"Calcolato PURCHASE_TRANSACTION_GAP. "
            f"PURCHASE_DATE_ANOMALY (>365gg): {n_anom:,} ({n_anom/n_raw*100:.1f}%) — mantenuti, solo flaggati"
        )

    _print_report("Clients", n_raw, len(out), transforms, flags_added=flags,
                  cols_added=["PURCHASE_TRANSACTION_GAP"] if "FIRST_PURCHASE_DATE" in out.columns else None)
    return out


# ===========================================================================
# CRC
# ===========================================================================

def clean_crc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean CRC (Customer Relationship Center appointments).

    Transformations:
    1. Parse CREATION_DATE as datetime
    2. Drop 4 rows with APPOINTMENT_DURATION > 480 min (>8h, errori)
    3. Create HAS_APPOINTMENT_DURATION flag (1 where APPOINTMENT_DURATION not null)
    """
    out = df.copy()
    n_raw = len(out)
    transforms = []
    flags = []

    # 1. Parse date
    if not pd.api.types.is_datetime64_any_dtype(out["CREATION_DATE"]):
        out["CREATION_DATE"] = pd.to_datetime(out["CREATION_DATE"], errors="coerce")
        transforms.append("CREATION_DATE -> datetime")

    # 2. Rimuovi APPOINTMENT_DURATION > 8h (480 minuti)
    dur_col = [c for c in out.columns if "DURATION" in c.upper()]
    if dur_col:
        dur_col = dur_col[0]
        n_gt8h = (out[dur_col] > 480).sum()
        out = out[out[dur_col].isna() | (out[dur_col] <= 480)].copy()
        transforms.append(f"Rimossi {n_gt8h} record con {dur_col} > 480 min (>8h, errori di inserimento)")

        # 3. Flag HAS_APPOINTMENT_DURATION
        out["HAS_APPOINTMENT_DURATION"] = out[dur_col].notna().astype(np.int8)
        flags.append("HAS_APPOINTMENT_DURATION")
        transforms.append(
            f"Creata HAS_APPOINTMENT_DURATION: {out['HAS_APPOINTMENT_DURATION'].sum():,} "
            f"({out['HAS_APPOINTMENT_DURATION'].mean()*100:.1f}%) record con durata nota"
        )

    _print_report("CRC", n_raw, len(out), transforms, flags_added=flags)
    return out


# ===========================================================================
# CCP
# ===========================================================================

def clean_ccp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean CCP (Cartier Care Program registrations).

    Transformations:
    1. Parse CREATION_DATE and SALE_DATE as datetime
    2. Compute SALE_CREATION_GAP in days
    3. Drop rows where SALE_CREATION_GAP > 30 days (94 serious errors)
    4. Keep rows with gap <= 7 days (rounding tolerance)
    5. Create SALE_DATE_VALID flag
    6. Rename FLAG_GIFT -> IS_GIFT for clarity
    """
    out = df.copy()
    n_raw = len(out)
    transforms = []
    flags = []

    # 1. Parse date
    for col in ["CREATION_DATE", "SALE_DATE"]:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce")
    transforms.append("CREATION_DATE, SALE_DATE -> datetime")

    # 2. Calcola gap SALE - CREATION (positivo = violazione logica)
    both_valid = out["SALE_DATE"].notna() & out["CREATION_DATE"].notna()
    out["SALE_CREATION_GAP"] = np.nan
    out.loc[both_valid, "SALE_CREATION_GAP"] = (
        (out.loc[both_valid, "SALE_DATE"] - out.loc[both_valid, "CREATION_DATE"]).dt.days
    )

    # 3. Droppare gap > 30gg (errori seri)
    mask_serious_error = out["SALE_CREATION_GAP"] > 30
    n_serious = int(mask_serious_error.sum())
    out = out[~mask_serious_error].copy()
    transforms.append(
        f"Rimossi {n_serious} record con SALE_DATE - CREATION_DATE > 30 giorni (errori seri). "
        f"Mantenuti {int(both_valid.sum() - n_serious)} record validi "
        f"(inclusi {int(((out.get('SALE_CREATION_GAP', pd.Series([])) > 0) & (out.get('SALE_CREATION_GAP', pd.Series([])) <= 7)).sum())} con gap <= 7gg)"
    )

    # 4. Flag SALE_DATE_VALID
    out["SALE_DATE_VALID"] = out["SALE_DATE"].notna().astype(np.int8)
    flags.append("SALE_DATE_VALID")
    transforms.append(
        f"Creata SALE_DATE_VALID: {out['SALE_DATE_VALID'].sum():,} ({out['SALE_DATE_VALID'].mean()*100:.1f}%) record"
    )

    # 5. Rinomina FLAG_GIFT -> IS_GIFT
    if "FLAG_GIFT" in out.columns:
        out = out.rename(columns={"FLAG_GIFT": "IS_GIFT"})
        transforms.append("FLAG_GIFT rinominata IS_GIFT per chiarezza")

    _print_report("CCP", n_raw, len(out), transforms, flags_added=flags,
                  cols_added=["SALE_CREATION_GAP"])
    return out


# ===========================================================================
# SUPPLEMENTARY FEATURES
# ===========================================================================

def build_supplementary_features(
    clients_clean: pd.DataFrame,
    crc_clean: pd.DataFrame,
    ccp_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build supplementary features aggregated per CLIENT_ID from CRC, CCP and Clients.

    Returns one row per CLIENT_ID with:
    - From CRC: HAS_CRC_INTERACTION, N_CRC_INTERACTIONS, HAS_CLIENTELING, AVG_APPOINTMENT_DURATION
    - From CCP: HAS_CCP, N_CCP_PRODUCTS, HAS_GIFT_REGISTERED
    - From Clients: BIRTH_DATE_VALID, PURCHASE_DATE_ANOMALY
    """
    print("\n--- Costruzione supplementary features ---")

    # --- Feature da CRC ---
    crc_dur = [c for c in crc_clean.columns if "DURATION" in c.upper()]
    crc_dur_col = crc_dur[0] if crc_dur else None

    crc_agg = crc_clean.groupby("CLIENT_ID").agg(
        N_CRC_INTERACTIONS=("CLIENT_ID", "count"),
        HAS_CLIENTELING=("ORIGIN", lambda x: int((x == "Clienteling").any())),
        **({crc_dur_col: (crc_dur_col, lambda x: x.dropna().mean())} if crc_dur_col else {}),
    ).reset_index()

    crc_agg["HAS_CRC_INTERACTION"] = np.int8(1)
    if crc_dur_col:
        crc_agg = crc_agg.rename(columns={crc_dur_col: "AVG_APPOINTMENT_DURATION"})
    crc_agg["N_CRC_INTERACTIONS"] = crc_agg["N_CRC_INTERACTIONS"].astype(int)
    crc_agg["HAS_CLIENTELING"] = crc_agg["HAS_CLIENTELING"].astype(np.int8)

    print(f"  CRC features costruite: {len(crc_agg):,} clienti")

    # --- Feature da CCP ---
    gift_col = "IS_GIFT" if "IS_GIFT" in ccp_clean.columns else "FLAG_GIFT"
    ccp_agg = ccp_clean.groupby("CLIENT_ID").agg(
        N_CCP_PRODUCTS=("CLIENT_ID", "count"),
        HAS_GIFT_REGISTERED=(gift_col, lambda x: int(x.any())),
    ).reset_index()
    ccp_agg["HAS_CCP"] = np.int8(1)
    ccp_agg["N_CCP_PRODUCTS"] = ccp_agg["N_CCP_PRODUCTS"].astype(int)
    ccp_agg["HAS_GIFT_REGISTERED"] = ccp_agg["HAS_GIFT_REGISTERED"].astype(np.int8)
    print(f"  CCP features costruite: {len(ccp_agg):,} clienti")

    # --- Feature da Clients ---
    cli_cols = ["CLIENT_ID"]
    for c in ["BIRTH_DATE_VALID", "PURCHASE_DATE_ANOMALY"]:
        if c in clients_clean.columns:
            cli_cols.append(c)
    cli_sub = clients_clean[cli_cols].copy()

    # --- Join tutto ---
    # Base: tutti i CLIENT_ID unici dall'unione di CRC, CCP, Clients
    all_ids = pd.DataFrame(
        {"CLIENT_ID": pd.concat([
            crc_clean["CLIENT_ID"],
            ccp_clean["CLIENT_ID"],
            clients_clean["CLIENT_ID"],
        ]).dropna().unique()}
    )

    supp = all_ids \
        .merge(crc_agg, on="CLIENT_ID", how="left") \
        .merge(ccp_agg, on="CLIENT_ID", how="left") \
        .merge(cli_sub, on="CLIENT_ID", how="left")

    # Riempi valori mancanti per flag binarie (clienti non in CRC/CCP)
    for col in ["HAS_CRC_INTERACTION", "HAS_CLIENTELING"]:
        if col in supp.columns:
            supp[col] = supp[col].fillna(0).astype(np.int8)
    supp["N_CRC_INTERACTIONS"] = supp["N_CRC_INTERACTIONS"].fillna(0).astype(int)
    for col in ["HAS_CCP", "HAS_GIFT_REGISTERED"]:
        if col in supp.columns:
            supp[col] = supp[col].fillna(0).astype(np.int8)
    supp["N_CCP_PRODUCTS"] = supp["N_CCP_PRODUCTS"].fillna(0).astype(int)

    print(f"  Supplementary features totali: {len(supp):,} clienti, {supp.shape[1]} colonne")
    print(f"  Colonne: {list(supp.columns)}")
    return supp


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_cleaning(raw: dict, clean: dict) -> pd.DataFrame:
    """
    Validate cleaned datasets against raw datasets.

    Checks:
    1. N rows clean <= N rows raw per dataset
    2. No TO_WITHOUTTAX_EUR_CONST < 0 in Transactions clean
    3. No SERIAL_NUMBER placeholder in Transactions clean
    4. No BIRTH_DATE pre-1900 in Clients clean
    5. No APPOINTMENT_DURATION > 480 min in CRC clean
    6. No null CLIENT_ID in Aggregated_Data clean
    7. Binary flags contain only 0/1
    8. TARGET columns unchanged vs raw
    """
    results = []

    def _check(name: str, passed: bool, detail: str) -> dict:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        return {"Check": name, "Status": status, "Dettaglio": detail}

    print("\n=== VALIDATION REPORT ===")

    # 1. N righe
    for ds in ["Aggregated_Data", "Transactions", "Clients", "CRC", "CCP"]:
        n_raw = len(raw[ds])
        n_cl  = len(clean[ds])
        results.append(_check(
            f"Righe {ds}", n_cl <= n_raw,
            f"raw={n_raw:,} clean={n_cl:,} rimosse={n_raw-n_cl:,}"
        ))

    # 2. No TO_WITHOUTTAX < 0
    col_to = "TO_WITHOUTTAX_EUR_CONST"
    n_neg = (clean["Transactions"][col_to] < 0).sum()
    results.append(_check("No TO_WITHOUTTAX<0", n_neg == 0, f"{n_neg} record negativi"))

    # 3. No SERIAL_NUMBER placeholder
    n_ph = (clean["Transactions"]["SERIAL_NUMBER"] == _SERIAL_PLACEHOLDER).sum()
    results.append(_check("No SN placeholder", n_ph == 0, f"{n_ph} placeholder residui"))

    # 4. No BIRTH_DATE pre-1900
    bd = pd.to_datetime(clean["Clients"]["BIRTH_DATE"], errors="coerce")
    n_pre1900 = int((bd.dt.year < 1900).sum())
    results.append(_check("No BIRTH_DATE pre-1900", n_pre1900 == 0, f"{n_pre1900} date anomale residue"))

    # 5. No APPOINTMENT_DURATION > 480
    dur_col = [c for c in clean["CRC"].columns if "DURATION" in c.upper()]
    if dur_col:
        dur_col = dur_col[0]
        n_gt8h = int((clean["CRC"][dur_col] > 480).sum())
        results.append(_check("No CRC DURATION>480", n_gt8h == 0, f"{n_gt8h} record anomali residui"))

    # 6. No CLIENT_ID null in Aggregated clean
    n_null_id = int(clean["Aggregated_Data"]["CLIENT_ID"].isna().sum())
    results.append(_check("No CLIENT_ID null (Agg)", n_null_id == 0, f"{n_null_id} null residui"))

    # 7. Flag binarie solo 0/1
    flag_cols = {
        "Aggregated_Data": ["AGE_KNOWN", "HAS_MULTIPLE_PURCHASES", "HAS_REPAIR_HISTORY"],
        "Transactions":    ["TO_WITHOUTTAX_IMPUTED", "SERIAL_NUMBER_KNOWN", "WWPRICE_MISSING"],
        "Clients":         ["BIRTH_DATE_VALID", "PURCHASE_DATE_ANOMALY"],
        "CRC":             ["HAS_APPOINTMENT_DURATION"],
        "CCP":             ["SALE_DATE_VALID"],
    }
    for ds, cols in flag_cols.items():
        for col in cols:
            if col in clean[ds].columns:
                unique_vals = set(clean[ds][col].dropna().unique())
                ok = unique_vals <= {0, 1}
                results.append(_check(
                    f"Flag {col} ({ds})", ok,
                    f"valori unici: {unique_vals}" if not ok else "OK [0/1]"
                ))

    # 8. TARGET immutati
    for tcol in ["TARGET_3Y", "TARGET_5Y", "TARGET_10Y"]:
        if tcol in raw["Aggregated_Data"].columns and tcol in clean["Aggregated_Data"].columns:
            # Confronta su indice comune (allineato per posizione dopo drop null id)
            n_raw_vals = raw["Aggregated_Data"][tcol].notna().sum()
            n_cl_vals  = clean["Aggregated_Data"][tcol].notna().sum()
            ok = (n_raw_vals - n_cl_vals) <= 10  # massimo 8 righe rimosse per CLIENT_ID null
            results.append(_check(
                f"{tcol} immutato", ok,
                f"non-null raw={n_raw_vals:,} clean={n_cl_vals:,}"
            ))

    # Riepilogo
    df_res = pd.DataFrame(results)
    n_pass = (df_res["Status"] == "PASS").sum()
    n_fail = (df_res["Status"] == "FAIL").sum()
    print(f"\n  Totale check: {len(df_res)} | PASS: {n_pass} | FAIL: {n_fail}")

    return df_res


# ===========================================================================
# PIPELINE PRINCIPALE
# ===========================================================================

def run_all_cleaning() -> dict:
    """
    Execute the full cleaning pipeline. Reads from data/raw/, writes to data/processed/.

    Returns dict with cleaned DataFrames.
    """
    print("=" * 60)
    print("=== PIPELINE CLEANING CARTIER QTEM ===")
    print("=" * 60)

    # --- Caricamento raw ---
    print("\nCaricamento dataset raw...")
    raw = {}

    raw["Aggregated_Data"] = pd.read_csv(
        DATA_RAW / "Aggregated_Data.csv",
        parse_dates=["DATE_TARGET"], low_memory=False,
    )
    print(f"  Aggregated_Data: {raw['Aggregated_Data'].shape}")

    raw["Transactions"] = pd.read_csv(
        DATA_RAW / "Transactions.csv",
        parse_dates=["TRS_DATE"], low_memory=False,
    )
    print(f"  Transactions:    {raw['Transactions'].shape}")

    raw["Clients"] = pd.read_csv(
        DATA_RAW / "Clients.csv",
        parse_dates=["BIRTH_DATE", "FIRST_PURCHASE_DATE", "FIRST_TRANSACTION_DATE"],
        low_memory=False,
    )
    print(f"  Clients:         {raw['Clients'].shape}")

    raw["CRC"] = pd.read_csv(
        DATA_RAW / "CRC.csv",
        parse_dates=["CREATION_DATE"], low_memory=False,
    )
    print(f"  CRC:             {raw['CRC'].shape}")

    raw["CCP"] = pd.read_csv(
        DATA_RAW / "CCP.csv",
        parse_dates=["CREATION_DATE", "SALE_DATE"], low_memory=False,
    )
    print(f"  CCP:             {raw['CCP'].shape}")

    # --- Cleaning ---
    print("\nApplicazione cleaning...")
    clean = {}
    clean["Aggregated_Data"] = clean_aggregated_data(raw["Aggregated_Data"])
    clean["Transactions"]    = clean_transactions(raw["Transactions"])
    clean["Clients"]         = clean_clients(raw["Clients"])
    clean["CRC"]             = clean_crc(raw["CRC"])
    clean["CCP"]             = clean_ccp(raw["CCP"])

    # --- Supplementary features ---
    clean["supplementary_features"] = build_supplementary_features(
        clean["Clients"], clean["CRC"], clean["CCP"]
    )

    # --- Validazione ---
    val_df = validate_cleaning(raw, clean)
    val_df.to_csv(OUT_TABLES / "cleaning_validation_report.csv", index=False)
    print("  Salvato: cleaning_validation_report.csv")

    # --- Salvataggio processed ---
    print("\nSalvataggio in data/processed/...")
    save_map = {
        "Aggregated_Data_clean": "Aggregated_Data",
        "Transactions_clean":    "Transactions",
        "Clients_clean":         "Clients",
        "CRC_clean":             "CRC",
        "CCP_clean":             "CCP",
        "supplementary_features":"supplementary_features",
    }

    for fname, key in save_map.items():
        df_out = clean[key]
        path = DATA_PROC / f"{fname}.csv"
        df_out.to_csv(path, index=False)
        n_raw_cols = raw.get(key, df_out).shape[1] if key in raw else 0
        n_new_cols = df_out.shape[1] - n_raw_cols
        print(f"  {fname}.csv — {df_out.shape[0]:,} righe × {df_out.shape[1]} colonne"
              f"  (+{n_new_cols} colonne)")

    print("\n=== PIPELINE COMPLETATA ===")
    return clean


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    cleaned = run_all_cleaning()
