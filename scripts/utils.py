"""
utils.py — Funzioni di utilità riutilizzabili per l'EDA Cartier QTEM.
Tutte le funzioni sono read-only: non modificano mai i dati originali.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --- Percorsi di progetto ---
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
OUT_TABLES = ROOT / "output" / "tables"
OUT_PLOTS = ROOT / "output" / "plots"

# Assicura che le cartelle di output esistano
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tipi attesi da Data Dictionary (estratti manualmente)
# ---------------------------------------------------------------------------
EXPECTED_TYPES = {
    "Aggregated_Data": {
        "CLIENT_ID":                    "ID (string hash)",
        "DATE_TARGET":                  "Date",
        "TARGET_3Y":                    "Continuous (float)",
        "TARGET_5Y":                    "Continuous (float)",
        "TARGET_10Y":                   "Continuous (float)",
        "AGE":                          "Continuous (float)",
        "SENIORITY":                    "Integer",
        "RESIDENCY_COUNTRY":            "Categorical",
        "RESIDENCY_MARKET":             "Categorical",
        "GENDER":                       "Categorical",
        "TO_FULL_HIST":                 "Continuous (float)",
        "TO_AVG_SPREAD":                "Continuous (float)",
        "TO_PAST_3Y":                   "Continuous (float)",
        "TO_PAST_3Y_6Y":                "Continuous (float)",
        "TO_OK_5K":                     "Continuous (float)",   # alias TO_0K_5K
        "TO_5K_10K":                    "Continuous (float)",
        "TO_10K_20K":                   "Continuous (float)",
        "TO_20K_50K":                   "Continuous (float)",
        "TO_MORE_10K":                  "Continuous (float)",   # alias TO_MORE_50K
        "TO_CRC":                       "Continuous (float)",
        "TO_WEB":                       "Continuous (float)",
        "TO_BTQ":                       "Continuous (float)",
        "TO_ACCESSORIES":               "Continuous (float)",
        "TO_FRAGRANCE":                 "Continuous (float)",
        "TO_JWL":                       "Continuous (float)",
        "TO_JWL_WAT_HE":               "Continuous (float)",
        "TO_OTHER_HE":                  "Continuous (float)",
        "TO_WAT":                       "Continuous (float)",
        "ALL_PURCHASED_PDT_CATEG":      "Categorical (list)",
        "ALL_PURCHASED_PDT_SUBCATEG":   "Categorical (list)",
        "ALL_PURCHASED_PDT_COLLECTION": "Categorical (list)",
        "ALL_PURCHASED_PDT_FUNCTION":   "Categorical (list)",
        "ALL_PURCHASED_PRICE_RANGE":    "Categorical (list)",
        "ALL_PURCHASED_DATES":          "Date (list)",
        "ALL_REPAIR_PDT_CATEG":         "Categorical (list)",
        "ALL_REPAIR_PDT_SUBCATEG":      "Categorical (list)",
        "ALL_REPAIR_PDT_COLLECTION":    "Categorical (list)",
        "ALL_REPAIR_PDT_FUNCTION":      "Categorical (list)",
        "ALL_REPAIR_PRICE_RANGE":       "Categorical (list)",
        "ALL_REPAIR_DATES":             "Date (list)",
    },
    "Transactions": {
        "CLIENT_ID":             "ID (string hash)",
        "ARTICLE_ID":            "ID",
        "CHANNEL":               "Categorical",
        "TRS_DATE":              "Date",
        "TRS_CATEG":             "Categorical",
        "ARTICLE_WWPRICE":       "Continuous (float)",
        "TO_WITHOUTTAX_EUR_CONST": "Continuous (float)",
        "QTY_PDT":               "Integer",
        "SERIAL_NUMBER":         "ID",
        "PDT_CATEG":             "Categorical",
        "PDT_SUBCATEG":          "Categorical",
        "PDT_FUNCTION":          "Categorical",
        "PDT_COLLECTION":        "Categorical",
        "FLAG_HE":               "Boolean",
    },
    "Clients": {
        "CLIENT_ID":                 "ID (string hash)",
        "COUNTRY_OF_RESIDENCE_CODE": "Categorical",
        "GENDER":                    "Categorical",
        "BIRTH_DATE":                "Date",
        "FIRST_PURCHASE_DATE":       "Date",
        "FIRST_TRANSACTION_DATE":    "Date",
        "CAN_BE_CONTACTED":          "Boolean",
        "CREATION_CHANNEL":          "Categorical",
    },
    "CRC": {
        "APPOINTMENT_ID":       "ID",
        "CLIENT_ID":            "ID (string hash)",
        "CREATION_DATE":        "Date",
        "APPOINTEMENT_DURATION": "Continuous (float/minutes)",
        "ORIGIN":               "Categorical",
    },
    "CCP": {
        "CLIENT_ID":     "ID (string hash)",
        "ARTICLE_ID":    "ID",
        "SERIAL_NUMBER": "ID",
        "CREATION_DATE": "Date",
        "SALE_DATE":     "Date",
        "FLAG_GIFT":     "Boolean",
    },
    "Articles": {
        "ARTICLE_ID":       "ID",
        "WORLD_PRICE":      "Continuous (float)",
        "FLAG_HE":          "Boolean",
        "FLAG_BRIDAL":      "Boolean",
        "FLAG_DIAMOND":     "Boolean",
        "PRODUCT_CATEGORY": "Categorical",
    },
    "savings_rate": {
        "Date":  "Date",
        "Value": "Continuous (float)",
    },
}


def _dtype_label(series: pd.Series) -> str:
    """Restituisce una label leggibile per il dtype di una Series pandas."""
    dtype = series.dtype
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime64"
    return "object"


def schema_check(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Valida lo schema di un DataFrame rispetto ai tipi attesi da Data Dictionary.

    Parametri
    ----------
    df           : DataFrame caricato da pd.read_csv / pd.read_excel
    dataset_name : nome del dataset (chiave in EXPECTED_TYPES)

    Ritorna
    -------
    DataFrame con colonne:
        Dataset | Colonna | Tipo rilevato | Tipo atteso | N. missing | % missing | Discrepanza | Note
    """
    expected = EXPECTED_TYPES.get(dataset_name, {})
    n_rows = len(df)
    records = []

    for col in df.columns:
        rilevato = _dtype_label(df[col])
        atteso = expected.get(col, "—")
        n_miss = int(df[col].isna().sum())
        pct_miss = round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Segnala discrepanza se il tipo rilevato non è compatibile con quello atteso
        discrepanza = ""
        if atteso != "—":
            atteso_lower = atteso.lower()
            if "date" in atteso_lower and rilevato not in ("datetime64", "object"):
                discrepanza = "TIPO ERRATO"
            elif "continuous" in atteso_lower and rilevato not in ("float64", "int64"):
                discrepanza = "TIPO ERRATO"
            elif "integer" in atteso_lower and rilevato not in ("int64", "float64"):
                discrepanza = "TIPO ERRATO"
            elif "boolean" in atteso_lower and rilevato not in ("bool", "object", "int64", "float64"):
                discrepanza = "TIPO ERRATO"

        # Note speciali per colonne chiave
        note = ""
        if col == "CLIENT_ID":
            sample = df[col].dropna().head(3).tolist()
            note = f"Campione: {sample}"
        elif "DATE" in col.upper() and rilevato == "object":
            note = "Stringa, non datetime — parsing mancante"

        records.append({
            "Dataset":       dataset_name,
            "Colonna":       col,
            "Tipo rilevato": rilevato,
            "Tipo atteso":   atteso,
            "N. missing":    n_miss,
            "% missing":     pct_miss,
            "Discrepanza":   discrepanza,
            "Note":          note,
        })

    return pd.DataFrame(records)


def missing_report(df: pd.DataFrame, dataset_name: str, threshold: float = 5.0) -> pd.DataFrame:
    """
    Genera un report delle colonne con valori mancanti sopra la soglia indicata.

    Parametri
    ----------
    df           : DataFrame caricato
    dataset_name : nome del dataset
    threshold    : soglia % (default 5%)

    Ritorna
    -------
    DataFrame ordinato per % missing decrescente, solo colonne >= threshold
    """
    n_rows = len(df)
    records = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct = round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0.0
        if pct >= threshold:
            records.append({
                "Dataset":    dataset_name,
                "Colonna":    col,
                "N. missing": n_miss,
                "% missing":  pct,
                "N. righe totali": n_rows,
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("% missing", ascending=False).reset_index(drop=True)
    return result


def load_all_datasets(data_dir: Path = DATA_RAW) -> dict[str, pd.DataFrame]:
    """
    Carica tutti i dataset dalla cartella raw e restituisce un dict {nome: DataFrame}.
    Aggregated_Data è caricato con parse_dates=['DATE_TARGET'].
    Non modifica mai i file originali.
    """
    datasets = {}

    print("=== CARICAMENTO DATASET ===")

    # Aggregated_Data — parsing con quoting standard (gestisce le liste CSV nelle ALL_* cols)
    print("  Caricamento Aggregated_Data.csv ...")
    datasets["Aggregated_Data"] = pd.read_csv(
        data_dir / "Aggregated_Data.csv",
        parse_dates=["DATE_TARGET"],
        low_memory=False,
    )
    print(f"    OK {datasets['Aggregated_Data'].shape}")

    # Transactions
    print("  Caricamento Transactions.csv ...")
    datasets["Transactions"] = pd.read_csv(
        data_dir / "Transactions.csv",
        parse_dates=["TRS_DATE"],
        low_memory=False,
    )
    print(f"    OK {datasets['Transactions'].shape}")

    # Clients
    print("  Caricamento Clients.csv ...")
    datasets["Clients"] = pd.read_csv(
        data_dir / "Clients.csv",
        parse_dates=["BIRTH_DATE", "FIRST_PURCHASE_DATE", "FIRST_TRANSACTION_DATE"],
        low_memory=False,
    )
    print(f"    OK {datasets['Clients'].shape}")

    # CRC
    print("  Caricamento CRC.csv ...")
    datasets["CRC"] = pd.read_csv(
        data_dir / "CRC.csv",
        parse_dates=["CREATION_DATE"],
        low_memory=False,
    )
    print(f"    OK {datasets['CRC'].shape}")

    # CCP
    print("  Caricamento CCP.csv ...")
    datasets["CCP"] = pd.read_csv(
        data_dir / "CCP.csv",
        parse_dates=["CREATION_DATE", "SALE_DATE"],
        low_memory=False,
    )
    print(f"    OK {datasets['CCP'].shape}")

    # Articles
    print("  Caricamento Articles.csv ...")
    datasets["Articles"] = pd.read_csv(
        data_dir / "Articles.csv",
        low_memory=False,
    )
    print(f"    OK {datasets['Articles'].shape}")

    # savings_rate
    print("  Caricamento savings_rate.csv ...")
    datasets["savings_rate"] = pd.read_csv(
        data_dir / "savings_rate.csv",
        parse_dates=["Date"],
        low_memory=False,
    )
    print(f"    OK {datasets['savings_rate'].shape}")

    print("=== TUTTI I DATASET CARICATI ===\n")
    return datasets


def print_finding(title: str, body: str = "") -> None:
    """Stampa un finding con header formattato."""
    sep = "=" * 60
    # Gestisce encoding Windows (cp1252 non supporta caratteri non-ASCII rari)
    def _safe(s: str) -> str:
        return s.encode("ascii", errors="replace").decode("ascii")
    print(f"\n{sep}")
    print(f"=== FINDING: {title.upper()} ===")
    print(sep)
    if body:
        print(_safe(body))


# ---------------------------------------------------------------------------
# Funzioni Fase 2 — Missing Values Analysis
# ---------------------------------------------------------------------------

def missing_temporal_pattern(
    df: pd.DataFrame,
    date_col: str = "DATE_TARGET",
    cols: list[str] | None = None,
    threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Calcola % missing per colonna × snapshot temporale.

    Parametri
    ----------
    df        : DataFrame panel (deve avere una colonna data/snapshot)
    date_col  : nome della colonna snapshot (default DATE_TARGET)
    cols      : lista colonne da analizzare; se None usa quelle con >threshold% missing
    threshold : soglia per selezionare le colonne automaticamente

    Ritorna
    -------
    DataFrame pivot: righe = colonne, colonne = snapshot, valori = % missing
    """
    if cols is None:
        pct_miss = df.isnull().mean() * 100
        cols = pct_miss[pct_miss >= threshold].index.tolist()
        if date_col in cols:
            cols.remove(date_col)

    records = []
    for col in cols:
        row = {"Colonna": col}
        for snap in sorted(df[date_col].unique()):
            mask = df[date_col] == snap
            pct = df.loc[mask, col].isnull().mean() * 100
            row[str(snap)[:10]] = round(pct, 1)
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Funzioni Fase 3 — Target Distribution
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """
    Calcola il coefficiente di Gini su valori > 0.
    Formula: G = (2 * sum(rank * x) / (n * sum(x))) - (n+1)/n
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return np.nan
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n)


def revenue_concentration(
    series: pd.Series,
    top_pcts: list[float] | None = None,
) -> dict[str, float]:
    """
    Calcola la quota di revenue generata dal top X% dei clienti.

    Parametri
    ----------
    series   : Series con valori target (inclusi gli zero)
    top_pcts : percentuali da calcolare (default [0.01, 0.05, 0.10, 0.20])

    Ritorna
    -------
    dict con chiavi 'top_1pct', 'top_5pct', ecc. e valori % revenue
    """
    if top_pcts is None:
        top_pcts = [0.01, 0.05, 0.10, 0.20]

    vals = series.dropna().values
    total = vals.sum()
    if total == 0:
        return {f"top_{int(p*100)}pct": np.nan for p in top_pcts}

    sorted_desc = np.sort(vals)[::-1]
    result = {}
    for p in top_pcts:
        n_top = max(1, int(len(sorted_desc) * p))
        share = sorted_desc[:n_top].sum() / total * 100
        result[f"top_{int(p*100)}pct"] = round(share, 2)
    return result


def target_stats(series: pd.Series, label: str = "") -> dict:
    """
    Statistiche complete per una variabile target (skewed, con massa a zero).

    Include: count, mean, std, percentili, skewness, kurtosis,
    pct_zero, pct_lt100, mean_med_ratio, gini.
    """
    from scipy import stats as scipy_stats

    vals = series.dropna()
    n = len(vals)
    if n == 0:
        return {"label": label, "count": 0}

    percs = np.percentile(vals, [25, 50, 75, 90, 95, 99, 99.9])
    med = percs[1]

    return {
        "label":           label,
        "count":           n,
        "mean":            round(vals.mean(), 2),
        "std":             round(vals.std(), 2),
        "min":             round(vals.min(), 2),
        "p25":             round(percs[0], 2),
        "p50":             round(med, 2),
        "p75":             round(percs[2], 2),
        "p90":             round(percs[3], 2),
        "p95":             round(percs[4], 2),
        "p99":             round(percs[5], 2),
        "p99_9":           round(percs[6], 2),
        "max":             round(vals.max(), 2),
        "skewness":        round(scipy_stats.skew(vals), 3),
        "kurtosis":        round(scipy_stats.kurtosis(vals), 3),
        "pct_zero":        round((vals == 0).mean() * 100, 2),
        "pct_lt100":       round((vals < 100).mean() * 100, 2),
        "mean_med_ratio":  round(vals.mean() / med, 2) if med > 0 else None,
        "gini":            round(gini_coefficient(vals.values), 4),
    }


# ---------------------------------------------------------------------------
# Funzioni Fase 4 — Leakage / Temporal Integrity
# ---------------------------------------------------------------------------

def check_date_list_leakage(
    df: pd.DataFrame,
    date_list_col: str,
    cutoff_col: str = "DATE_TARGET",
    n_sample: int = 200,
) -> dict[str, int]:
    """
    Verifica che le date in una colonna lista (es. ALL_PURCHASED_DATES)
    siano tutte <= cutoff_col per ogni snapshot.

    Parametri
    ----------
    df            : DataFrame panel
    date_list_col : colonna con date separate da virgola
    cutoff_col    : colonna del cutoff (DATE_TARGET)
    n_sample      : righe da campionare per snapshot (per velocita)

    Ritorna
    -------
    dict con chiavi 'total_checked', 'total_violations'
    """
    total_ok = 0
    total_viol = 0

    for snap in sorted(df[cutoff_col].unique()):
        snap_date = pd.Timestamp(snap).date()
        sample = (
            df[(df[cutoff_col] == snap) & df[date_list_col].notna()][date_list_col]
            .head(n_sample)
        )
        if len(sample) == 0:
            continue
        all_dates_raw = sample.str.split(",").explode().str.strip()
        parsed = pd.to_datetime(all_dates_raw, errors="coerce").dt.date.dropna()
        total_viol += int((parsed > snap_date).sum())
        total_ok += int((parsed <= snap_date).sum())

    return {"total_checked": total_ok + total_viol, "total_violations": total_viol}


# ---------------------------------------------------------------------------
# Funzioni Fase 5 — Quality Analysis
# ---------------------------------------------------------------------------

def transactions_temporal_coverage(
    trs: pd.DataFrame,
    date_targets: list,
    trs_date_col: str = "TRS_DATE",
) -> pd.DataFrame:
    """
    Per ogni snapshot DATE_TARGET conta transazioni safe (<=) e a rischio leakage (>).

    Ritorna
    -------
    DataFrame con colonne: DATE_TARGET, TRS safe, TRS leakage, Total, % leakage
    """
    rows = []
    for snap in sorted(date_targets):
        safe  = int((trs[trs_date_col] <= snap).sum())
        leak  = int((trs[trs_date_col] >  snap).sum())
        total = safe + leak
        rows.append({
            "DATE_TARGET":            str(snap)[:10],
            "TRS safe (<= DT)":       safe,
            "TRS leakage (> DT)":     leak,
            "Total":                  total,
            "% leakage":              round(leak / total * 100, 2) if total > 0 else 0,
        })
    return pd.DataFrame(rows)


def serial_number_analysis(
    trs: pd.DataFrame,
    sn_col: str = "SERIAL_NUMBER",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Analizza la distribuzione di NULL e del placeholder per SERIAL_NUMBER.

    Parametri
    ----------
    trs        : DataFrame Transactions
    sn_col     : nome colonna SERIAL_NUMBER
    group_cols : colonne di raggruppamento (default: TRS_CATEG, CHANNEL, CATEG)

    Ritorna
    -------
    DataFrame con N e % di NULL e placeholder per ogni valore nei group_cols
    """
    if group_cols is None:
        group_cols = ["TRS_CATEG", "CHANNEL", "CATEG"]

    placeholder = trs[sn_col].value_counts().idxmax() if trs[sn_col].notna().any() else None
    rows = []
    for col in group_cols:
        for val in trs[col].dropna().unique():
            mask_val  = trs[col] == val
            total_val = int(mask_val.sum())
            null_val  = int((mask_val & trs[sn_col].isna()).sum())
            ph_val    = int((mask_val & (trs[sn_col] == placeholder)).sum()) if placeholder else 0
            rows.append({
                "Dimensione":         col,
                "Valore":             val,
                "N totale":           total_val,
                "N NULL SN":          null_val,
                "% NULL SN":          round(null_val / total_val * 100, 1) if total_val else 0,
                "N Placeholder SN":   ph_val,
                "% Placeholder SN":   round(ph_val / total_val * 100, 1) if total_val else 0,
            })
    return pd.DataFrame(rows)


def referential_integrity_matrix(id_sets: dict[str, set]) -> pd.DataFrame:
    """
    Calcola la matrice di overlap CLIENT_ID tra dataset.

    Parametri
    ----------
    id_sets : dict {nome_dataset: set(CLIENT_ID)}

    Ritorna
    -------
    DataFrame con righe: Dataset_A, Dataset_B, N intersezione, N solo A/B, % coverage
    """
    names = list(id_sets.keys())
    rows = []
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i >= j:
                continue
            set_a = id_sets[name_a]
            set_b = id_sets[name_b]
            inter = set_a & set_b
            only_a = set_a - set_b
            only_b = set_b - set_a
            rows.append({
                "Dataset_A":       name_a,
                "Dataset_B":       name_b,
                "N_A":             len(set_a),
                "N_B":             len(set_b),
                "N_intersection":  len(inter),
                "N_only_A":        len(only_a),
                "N_only_B":        len(only_b),
                "pct_A_in_B":      round(len(inter) / len(set_a) * 100, 1) if set_a else 0,
                "pct_B_in_A":      round(len(inter) / len(set_b) * 100, 1) if set_b else 0,
            })
    return pd.DataFrame(rows)


def appointment_duration_mar_test(
    crc: pd.DataFrame,
    dur_col: str,
    origin_col: str = "ORIGIN",
) -> pd.DataFrame:
    """
    Calcola % missing di APPOINTMENT_DURATION per ogni valore di ORIGIN.
    Ritorna DataFrame ordinato per % missing decrescente + diagnosi MAR/MNAR.

    La diagnosi è basata sulla varianza inter-gruppo:
    - std > 10 punti percentuali → MAR (missing correlato a ORIGIN)
    - std <= 10 → MNAR (distribuzione uniforme indipendentemente da ORIGIN)
    """
    origin_totals = crc.groupby(origin_col).size().rename("n_totale")
    dur_null = crc[crc[dur_col].isna()].groupby(origin_col).size().rename("n_missing")
    result = pd.concat([origin_totals, dur_null], axis=1).fillna(0).reset_index()
    result["n_missing"] = result["n_missing"].astype(int)
    result["pct_missing"] = (result["n_missing"] / result["n_totale"] * 100).round(1)
    result = result.sort_values("pct_missing", ascending=False).reset_index(drop=True)

    std_val = result["pct_missing"].std()
    result["diagnosi"] = "MAR" if std_val > 10 else "MNAR"
    result["std_pct_missing"] = round(std_val, 1)
    return result


def classify_columns_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restituisce la classificazione temporale delle colonne di Aggregated_Data.
    Utile per esportare la tabella in output/tables/.
    """
    classification = OUT_TABLES.parent / "output" / "tables" / "column_temporal_classification.csv"
    if classification.exists():
        return pd.read_csv(classification)

    # Fallback: classi hardcoded (vedi _run_phase4a.py per la lista completa)
    records = []
    targets = {"TARGET_3Y", "TARGET_5Y", "TARGET_10Y"}
    ids = {"CLIENT_ID", "DATE_TARGET"}
    statics = {"GENDER", "RESIDENCY_COUNTRY", "RESIDENCY_MARKET"}
    risk = {"RECENCY", "ALL_PURCHASED_DATES", "ALL_REPAIR_DATES"}

    for col in df.columns:
        if col in targets:
            cat = "TARGET"
        elif col in ids:
            cat = "TEMPORAL_ID"
        elif col in statics:
            cat = "STATIC"
        elif col in risk:
            cat = "RISK_LEAKAGE"
        else:
            cat = "SAFE_FEATURE"
        records.append({"Colonna": col, "Categoria": cat})

    return pd.DataFrame(records)
