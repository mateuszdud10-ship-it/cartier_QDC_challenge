"""
Costruisce la Decision Map completa da tutti i finding delle fasi 1-7.
Salva in output/tables/decision_map.csv.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
from utils import OUT_TABLES

rows = [
    # ─── AGGREGATED_DATA ───────────────────────────────────────────────────
    {
        "Dataset": "Aggregated_Data", "Problema": "CLIENT_ID null",
        "N record": 8, "% totale": "~0%",
        "Decisione": "DROP righe",
        "Motivazione": "CLIENT_ID e la chiave primaria — senza di essa la riga e inutilizzabile",
        "Impatto modello": "Trascurabile (8 righe su 1.5M)",
    },
    {
        "Dataset": "Aggregated_Data", "Problema": "AGE: 70.8% missing (MAR temporale)",
        "N record": 1075042, "% totale": "70.8%",
        "Decisione": "Creare flag AGE_KNOWN (0/1). Non imputare.",
        "Motivazione": "Missing decresce da 94.6% (2006) a 63.2% (2021). Imputare introdurrebbe bias temporale. La flag preserva il segnale informativo.",
        "Impatto modello": "AGE usabile solo su snapshot 2021 con AGE_KNOWN=1 + imputazione mediana per mercato/genere in feature engineering",
    },
    {
        "Dataset": "Aggregated_Data", "Problema": "STDDEV/AVG/MIN_TIMELAPSE_TRS: 84-95% missing strutturale",
        "N record": 1276948, "% totale": "~85%",
        "Decisione": "Creare flag HAS_MULTIPLE_PURCHASES (0/1). Non imputare valori originali.",
        "Motivazione": "Missing strutturale: presenti solo se NB_TRS > 1. La flag e informativa di per se (segnale di loyalty).",
        "Impatto modello": "HAS_MULTIPLE_PURCHASES e feature diretta per la Parte 1 del Two-Part Model",
    },
    {
        "Dataset": "Aggregated_Data", "Problema": "ALL_REPAIR_*: missing strutturale",
        "N record": 0, "% totale": "varia",
        "Decisione": "Creare flag HAS_REPAIR_HISTORY (0/1). Non imputare.",
        "Motivazione": "Missing strutturale: presenti solo se cliente ha riparazioni. La flag cattura il profilo 'cliente con prodotti'.",
        "Impatto modello": "Segnale di retention — cliente con repair ha prodotto attivo e probabilmente torni",
    },
    {
        "Dataset": "Aggregated_Data", "Problema": "TARGET_3Y=0: 95.2% su snapshot 2021",
        "N record": 392652, "% totale": "95.2%",
        "Decisione": "Nessuna azione. Two-Part Model gestisce la massa a zero esplicitamente.",
        "Motivazione": "Distribuzione power-law attesa. La massa a zero e il fenomeno da modellare, non un problema di qualita.",
        "Impatto modello": "CRITICO — definisce la Parte 1 (classificatore P(TARGET>0))",
    },
    {
        "Dataset": "Aggregated_Data", "Problema": "TO_FULL_HIST=0 con TARGET>0",
        "N record": 0, "% totale": "0%",
        "Decisione": "Nessuna azione. Dataset coerente.",
        "Motivazione": "Zero violazioni confermate in Fase 7B.",
        "Impatto modello": "Nessuno",
    },

    # ─── TRANSACTIONS ───────────────────────────────────────────────────────
    {
        "Dataset": "Transactions", "Problema": "TO_WITHOUTTAX_EUR_CONST < 0 (4 righe)",
        "N record": 4, "% totale": "0.0004%",
        "Decisione": "DROP righe",
        "Motivazione": "Rettifiche su Repair. Impatto trascurabile. Non codificano resi (che sarebbero in TRS_CATEG=Sale).",
        "Impatto modello": "Trascurabile",
    },
    {
        "Dataset": "Transactions", "Problema": "TO_WITHOUTTAX_EUR_CONST null (422 righe)",
        "N record": 422, "% totale": "0.046%",
        "Decisione": "IMPUTA: Sale->ARTICLE_WWPRICE, Repair->0. Flag TO_WITHOUTTAX_IMPUTED.",
        "Motivazione": "WWPRICE e la miglior stima per vendite; 0 e conservativo per riparazioni (potrebbe essere servizio gratuito).",
        "Impatto modello": "Minimo. Flag permette di escludere imputati dal training se necessario.",
    },
    {
        "Dataset": "Transactions", "Problema": "SERIAL_NUMBER null (97.221 = 10.6%)",
        "N record": 97221, "% totale": "10.6%",
        "Decisione": "Mantenere null. Creare flag SERIAL_NUMBER_KNOWN (0/1).",
        "Motivazione": "Vendite senza serial tracciato sono legittime (accessori, fragranze). La flag e informativa.",
        "Impatto modello": "SERIAL_NUMBER non sara feature diretta — usata solo come indicatore di tipo prodotto",
    },
    {
        "Dataset": "Transactions", "Problema": "SERIAL_NUMBER placeholder '4e2e3377...' (16.741 = 1.8%)",
        "N record": 16741, "% totale": "1.8%",
        "Decisione": "Sostituire con NaN. Incluso in SERIAL_NUMBER_KNOWN=0.",
        "Motivazione": "Il placeholder maschera un null reale. Trattarlo come null evita contaminazione nei join su SERIAL_NUMBER.",
        "Impatto modello": "Minimo",
    },
    {
        "Dataset": "Transactions", "Problema": "ARTICLE_WWPRICE=0 con TO_WITHOUTTAX>0 — Repair (3.386)",
        "N record": 3386, "% totale": "0.37%",
        "Decisione": "MANTENERE. Flag WWPRICE_MISSING=1.",
        "Motivazione": "LEGITTIMO: costo di servizio riparazione. TO_WITHOUTTAX e il valore fatturato corretto. WWPRICE non e il prezzo del servizio ma del prodotto nel catalogo.",
        "Impatto modello": "Da escludere dai calcoli di prezzo medio ma includere nel TO aggregato",
    },
    {
        "Dataset": "Transactions", "Problema": "ARTICLE_WWPRICE=0 con TO_WITHOUTTAX>0 — Sale (20.158)",
        "N record": 20158, "% totale": "2.2%",
        "Decisione": "MANTENERE. Flag WWPRICE_MISSING=1.",
        "Motivazione": "Articoli legacy/custom non catalogati. Il TO_WITHOUTTAX e valido. Droppare perderebbe informazione di spend.",
        "Impatto modello": "Inclusi nel TO aggregato per cliente. Esclusi da feature basate su WWPRICE.",
    },
    {
        "Dataset": "Transactions", "Problema": "TRS_DATE > DATE_TARGET (leakage per snapshot)",
        "N record": 63683, "% totale": "6.9% (post-2021)",
        "Decisione": "NON filtrare qui. Filtrare in feature engineering per ogni snapshot (TRS_DATE <= DATE_TARGET).",
        "Motivazione": "Il filtro e snapshot-specifico. Applicarlo in cleaning eliminerebbe transazioni legittime per snapshot storici.",
        "Impatto modello": "CRITICO — applicare in feature engineering. Regola: TRS_DATE <= DATE_TARGET per ogni snapshot.",
    },
    {
        "Dataset": "Transactions", "Problema": "QTY_PDT > 10 (56 righe)",
        "N record": 56, "% totale": "0.006%",
        "Decisione": "MANTENERE. Nessuna azione.",
        "Motivazione": "Acquisti multipli dello stesso articolo (es. accessori) sono legittimi. 56 righe su 916k e trascurabile.",
        "Impatto modello": "Trascurabile",
    },

    # ─── CLIENTS ────────────────────────────────────────────────────────────
    {
        "Dataset": "Clients", "Problema": "BIRTH_DATE pre-1900 (1.891)",
        "N record": 1891, "% totale": "0.46%",
        "Decisione": "NULLIFY (set a NaT). Flag BIRTH_DATE_VALID=0.",
        "Motivazione": "Date impossibili (placeholder 1804-01-01). Trattarle come missing e corretto.",
        "Impatto modello": "Minimo. AGE non sara usata direttamente (vedi AGE in Aggregated).",
    },
    {
        "Dataset": "Clients", "Problema": "BIRTH_DATE eta < 18 anni (421)",
        "N record": 421, "% totale": "0.10%",
        "Decisione": "NULLIFY. Flag BIRTH_DATE_VALID=0.",
        "Motivazione": "Eta incompatibile con cliente Cartier. Probabili errori di inserimento.",
        "Impatto modello": "Minimo",
    },
    {
        "Dataset": "Clients", "Problema": "BIRTH_DATE eta > 100 anni (1.993)",
        "N record": 1993, "% totale": "0.48%",
        "Decisione": "NULLIFY. Flag BIRTH_DATE_VALID=0.",
        "Motivazione": "Eta implausibile per cliente attivo. Quasi certamente placeholder o errore.",
        "Impatto modello": "Minimo",
    },
    {
        "Dataset": "Clients", "Problema": "FIRST_PURCHASE vs FIRST_TRANSACTION gap > 365gg (7.243)",
        "N record": 7243, "% totale": "1.8%",
        "Decisione": "MANTENERE. Flag PURCHASE_DATE_ANOMALY. Calcola gap in giorni.",
        "Motivazione": "Potrebbe indicare clienti con acquisti offline non tracciati digitalmente. Non c'e motivo certo per droppare.",
        "Impatto modello": "PURCHASE_DATE_ANOMALY puo essere feature per identificare clienti con storico parziale",
    },

    # ─── CRC ────────────────────────────────────────────────────────────────
    {
        "Dataset": "CRC", "Problema": "APPOINTMENT_DURATION missing (86.1% — MAR)",
        "N record": 132576, "% totale": "86.1%",
        "Decisione": "Creare flag HAS_APPOINTMENT_DURATION (0/1). Non imputare.",
        "Motivazione": "MAR: missing correlato a ORIGIN (Phone/Email = 100% missing, CRC = 0%). Imputare introdurrebbe bias. La flag cattura il segnale.",
        "Impatto modello": "HAS_APPOINTMENT_DURATION come feature binaria in supplementary_features",
    },
    {
        "Dataset": "CRC", "Problema": "APPOINTMENT_DURATION > 480 min (4 righe)",
        "N record": 4, "% totale": "0.003%",
        "Decisione": "DROP righe",
        "Motivazione": "Appuntamenti > 8 ore impossibili operativamente. Errori di inserimento certi.",
        "Impatto modello": "Trascurabile",
    },

    # ─── CCP ────────────────────────────────────────────────────────────────
    {
        "Dataset": "CCP", "Problema": "SALE_DATE > CREATION_DATE <= 7gg (121)",
        "N record": 121, "% totale": "0.27%",
        "Decisione": "TOLLERARE. Non droppare.",
        "Motivazione": "Differenze di pochi giorni probabilmente dovute ad arrotondamento o timezone. Il dato rimane utilizzabile.",
        "Impatto modello": "Minimo",
    },
    {
        "Dataset": "CCP", "Problema": "SALE_DATE > CREATION_DATE > 30gg (94)",
        "N record": 94, "% totale": "0.21%",
        "Decisione": "DROP righe",
        "Motivazione": "Differenze > 30 giorni logicamente impossibili (un prodotto non puo essere venduto dopo essere stato registrato). Errori certi.",
        "Impatto modello": "Trascurabile",
    },
    {
        "Dataset": "CCP", "Problema": "SALE_DATE missing (60.8%)",
        "N record": 26932, "% totale": "60.8%",
        "Decisione": "Creare flag SALE_DATE_VALID (0/1). Non imputare.",
        "Motivazione": "Missing simile tra FLAG_GIFT True (65.1%) e False (60.2%) — non fortemente MAR. La flag e informativa.",
        "Impatto modello": "SALE_DATE_VALID come feature in supplementary_features",
    },

    # ─── ARTICLES ───────────────────────────────────────────────────────────
    {
        "Dataset": "Articles", "Problema": "WORLD_PRICE < 1 EUR (20 articoli — tutti ProductCategory_7)",
        "N record": 20, "% totale": "0.03%",
        "Decisione": "MANTENERE. Non modificare.",
        "Motivazione": "ProductCategory_7 non appare in Transactions — questi articoli non contribuiscono ai calcoli. Impatto nullo.",
        "Impatto modello": "Nessuno (categoria non in Transactions)",
    },
    {
        "Dataset": "Articles", "Problema": "ARTICLE_ID orfani in Transactions (5.378 = 18.6%)",
        "N record": 5378, "% totale": "18.6% di TRS ARTICLE_ID",
        "Decisione": "Usare LEFT JOIN (non INNER JOIN) in feature engineering tra Transactions e Articles.",
        "Motivazione": "Articoli dismessi o codici legacy. Il valore di spend e presente in Transactions — non va perso con un inner join.",
        "Impatto modello": "WWPRICE sara NULL per orfani — gestito da WWPRICE_MISSING flag",
    },
]

dm = pd.DataFrame(rows)
dm.to_csv(OUT_TABLES / "decision_map.csv", index=False)
print(f"Decision Map salvata: {len(dm)} decisioni in output/tables/decision_map.csv")
print("\n=== DECISION MAP COMPLETA ===")
print(dm[["Dataset", "Problema", "N record", "Decisione"]].to_string(index=False))
