"""
build_fe_map.py — Costruisce la mappa delle feature ingegnerizzabili per il feature engineering.
Salva in output/tables/feature_engineering_map.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_TABLES = ROOT / "output" / "tables"

feature_map = [
    # A: RFM da Transactions
    {"Categoria":"A_RFM_Transactions","Nome":"RECENCY_DAYS","Logica":"Giorni dall'ultimo acquisto a DATE_TARGET","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - filtro TRS_DATE<=DATE_TARGET obbligatorio"},
    {"Categoria":"A_RFM_Transactions","Nome":"N_TRANSACTIONS","Logica":"COUNT(TRS_DATE <= DATE_TARGET) per cliente","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - filtro temporale"},
    {"Categoria":"A_RFM_Transactions","Nome":"TOTAL_SPEND","Logica":"SUM(TO_WITHOUTTAX_EUR_CONST) con TRS_DATE<=DATE_TARGET","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - filtro temporale"},
    {"Categoria":"A_RFM_Transactions","Nome":"AVG_SPEND_PER_TRS","Logica":"TOTAL_SPEND / N_TRANSACTIONS","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"SPEND_PAST_3Y","Logica":"SUM(TO_WITHOUTTAX) in [DATE_TARGET-3Y, DATE_TARGET]","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - finestra temporale esplicita"},
    {"Categoria":"A_RFM_Transactions","Nome":"SPEND_3Y_6Y","Logica":"SUM(TO_WITHOUTTAX) in [DATE_TARGET-6Y, DATE_TARGET-3Y)","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"SPEND_TREND","Logica":"SPEND_PAST_3Y / max(SPEND_3Y_6Y, 1) - ratio crescita spend","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"N_SALE_TRS","Logica":"COUNT(TRS_CATEG=Sale, TRS_DATE<=DATE_TARGET)","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"N_REPAIR_TRS","Logica":"COUNT(TRS_CATEG=Repair, TRS_DATE<=DATE_TARGET)","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"REPAIR_RATIO","Logica":"N_REPAIR_TRS / N_TRANSACTIONS","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"CHANNEL_BOUTIQUE_RATIO","Logica":"COUNT(CHANNEL=Boutique) / N_TRANSACTIONS","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"A_RFM_Transactions","Nome":"DAYS_BETWEEN_TRS_AVG","Logica":"Media gap giorni tra transazioni consecutive per cliente","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - disponibile anche in Aggregated come AVG_TIMELAPSE_PER_TRS"},
    # B: Feature gia in Aggregated
    {"Categoria":"B_Aggregated_Numeriche","Nome":"TO_FULL_HIST","Logica":"Spesa totale storica - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"TO_PAST_3Y","Logica":"Spesa ultimi 3 anni - gia in Aggregated (73% zero)","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"TO_PAST_3Y_6Y","Logica":"Spesa periodo 3-6 anni fa - gia in Aggregated (74% zero)","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"RECENCY","Logica":"Recenza in giorni - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"SENIORITY","Logica":"Anni da primo acquisto - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"NB_TRS_FULL_HIST","Logica":"N transazioni storiche - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"AVG_PRICE_PER_PDT","Logica":"Prezzo medio per prodotto - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"MAX_PRICE_PER_PDT","Logica":"Prezzo massimo pagato - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"TO_BTQ / TO_WEB / TO_CRC","Logica":"Spesa per canale - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"TO_JWL / TO_WAT / TO_ACCESSORIES","Logica":"Spesa per categoria - gia in Aggregated","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"NB_TRS_2Y_IN_A_ROW","Logica":"N periodi con acquisti consecutivi per anno","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Numeriche","Nome":"QTY_PDT_FULL_HIST","Logica":"Quantita totale prodotti acquistati","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Flags","Nome":"AGE_KNOWN","Logica":"Flag: AGE e disponibile","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Flags","Nome":"HAS_MULTIPLE_PURCHASES","Logica":"Flag: NB_TRS > 1","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Flags","Nome":"HAS_REPAIR_HISTORY","Logica":"Flag: almeno una riparazione storica","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Categoriche","Nome":"GENDER","Logica":"Genere cliente - encoding OHE","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Categoriche","Nome":"RESIDENCY_MARKET","Logica":"Mercato di residenza - encoding OHE o target encoding","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"B_Aggregated_Categoriche","Nome":"AGE (imputato snap 2021)","Logica":"AGE solo snapshot 2021 + imputazione mediana per mercato/genere","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"VERIFICARE - solo snap 2021, MAR"},
    # C: Feature da Articles
    {"Categoria":"C_Articles_Join","Nome":"AVG_ARTICLE_PRICE","Logica":"Mean(WORLD_PRICE) per articoli acquistati - left join Articles","Sorgente":"Transactions_clean + Articles","Rischio_Leakage":"SAFE - left join, WWPRICE_MISSING per orfani"},
    {"Categoria":"C_Articles_Join","Nome":"MAX_ARTICLE_PRICE","Logica":"Max(WORLD_PRICE) per articoli acquistati","Sorgente":"Transactions_clean + Articles","Rischio_Leakage":"SAFE"},
    {"Categoria":"C_Articles_Join","Nome":"FLAG_HE_RATIO","Logica":"Proporzione transazioni con FLAG_HE=1","Sorgente":"Transactions_clean (FLAG_HE gia presente)","Rischio_Leakage":"SAFE - FLAG_HE gia in Transactions_clean"},
    {"Categoria":"C_Articles_Join","Nome":"FLAG_BRIDAL_RATIO","Logica":"Proporzione acquisti Bridal - join Articles su ARTICLE_ID","Sorgente":"Transactions_clean + Articles","Rischio_Leakage":"SAFE"},
    {"Categoria":"C_Articles_Join","Nome":"FLAG_DIAMOND_RATIO","Logica":"Proporzione acquisti Diamond","Sorgente":"Transactions_clean + Articles","Rischio_Leakage":"SAFE"},
    {"Categoria":"C_Articles_Join","Nome":"N_DISTINCT_CATEGORIES","Logica":"COUNT(DISTINCT CATEG) per cliente - da Transactions_clean","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - CATEG gia in Transactions"},
    # D: Supplementary features
    {"Categoria":"D_Supplementary","Nome":"HAS_CRC_INTERACTION","Logica":"Cliente ha interagito con CRC (coverage 6.8% snap 2021)","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - CRC non filtrata per snapshot; usare CREATION_DATE<=DATE_TARGET"},
    {"Categoria":"D_Supplementary","Nome":"N_CRC_INTERACTIONS","Logica":"N interazioni CRC totali","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale necessario"},
    {"Categoria":"D_Supplementary","Nome":"HAS_CLIENTELING","Logica":"Almeno un appuntamento Clienteling","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale"},
    {"Categoria":"D_Supplementary","Nome":"AVG_APPOINTMENT_DURATION","Logica":"Durata media appuntamenti (ORIGIN in CRC/Clienteling/Web)","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale"},
    {"Categoria":"D_Supplementary","Nome":"HAS_CCP","Logica":"Cliente ha registrato almeno un prodotto in CCP (coverage 1.4%)","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale; usare come segnale binario"},
    {"Categoria":"D_Supplementary","Nome":"N_CCP_PRODUCTS","Logica":"N prodotti registrati in CCP","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale"},
    {"Categoria":"D_Supplementary","Nome":"HAS_GIFT_REGISTERED","Logica":"Cliente ha ricevuto/registrato un regalo","Sorgente":"supplementary_features","Rischio_Leakage":"VERIFICARE - filtro temporale"},
    {"Categoria":"D_Supplementary","Nome":"BIRTH_DATE_VALID","Logica":"BIRTH_DATE e valida (non anomala)","Sorgente":"supplementary_features","Rischio_Leakage":"SAFE"},
    {"Categoria":"D_Supplementary","Nome":"PURCHASE_DATE_ANOMALY","Logica":"Gap FIRST_PURCHASE vs FIRST_TRANSACTION > 365gg","Sorgente":"supplementary_features","Rischio_Leakage":"SAFE"},
    # E: Feature temporali/stagionali
    {"Categoria":"E_Temporali_Stagionali","Nome":"MONTH_LAST_PURCHASE","Logica":"Mese (1-12) dell'ultimo acquisto prima di DATE_TARGET","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE - filtro TRS_DATE<=DATE_TARGET"},
    {"Categoria":"E_Temporali_Stagionali","Nome":"HOLIDAY_PURCHASE_RATIO","Logica":"Quota acquisti in febbraio (Valentine) o dicembre (Natale)","Sorgente":"Transactions_clean","Rischio_Leakage":"SAFE"},
    {"Categoria":"E_Temporali_Stagionali","Nome":"SNAPSHOT_YEAR","Logica":"Anno snapshot (2006/2009/.../2021) - cattura trend temporale dataset","Sorgente":"Aggregated_Data_clean (DATE_TARGET)","Rischio_Leakage":"SAFE - e il cutoff stesso"},
    {"Categoria":"E_Temporali_Stagionali","Nome":"SPEND_TREND_LOG","Logica":"log1p(SPEND_PAST_3Y) - log1p(SPEND_3Y_6Y) trend in log-space","Sorgente":"Aggregated_Data_clean o Transactions","Rischio_Leakage":"SAFE"},
    # ESCLUDI
    {"Categoria":"ESCLUDI","Nome":"TARGET_5Y / TARGET_10Y come feature","Logica":"Variabili target - NON usare come feature","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"ESCLUDI - leakage diretto"},
    {"Categoria":"ESCLUDI","Nome":"MAX_PRICE_IN_BTQ","Logica":"100% zero - colonna costante, nessun segnale","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"ESCLUDI - zero variance"},
    {"Categoria":"ESCLUDI","Nome":"NB_TRS_BTQ","Logica":"100% zero - colonna costante","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"ESCLUDI - zero variance"},
    {"Categoria":"ESCLUDI","Nome":"TO_OTHER_HE","Logica":"99.99% zero - near-zero variance","Sorgente":"Aggregated_Data_clean","Rischio_Leakage":"ESCLUDI - near-zero variance"},
]

df_map = pd.DataFrame(feature_map)
df_map.to_csv(OUT_TABLES / "feature_engineering_map.csv", index=False)
print(f"Salvato: {OUT_TABLES / 'feature_engineering_map.csv'} ({len(df_map)} feature)")
print()
print(df_map[["Categoria","Nome","Rischio_Leakage"]].to_string(index=False))
