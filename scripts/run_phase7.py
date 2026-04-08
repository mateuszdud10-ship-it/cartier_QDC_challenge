"""
Script Fase 7 — Outlier Analysis.
Eseguire dalla root: python3 scripts/run_phase7.py
"""

import sys, warnings
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils import load_all_datasets, print_finding, OUT_TABLES

dfs = load_all_datasets()
agg = dfs['Aggregated_Data']
trs = dfs['Transactions']
cli = dfs['Clients']
crc = dfs['CRC']
ccp = dfs['CCP']
art = dfs['Articles']

agg_2021 = agg[agg['DATE_TARGET'].dt.year == 2021].copy()
snap_2021 = pd.Timestamp('2021-01-01')
N = len(agg_2021)
print(f"Snapshot 2021: {N:,} righe")


# ===========================================================================
# 7A — Target outliers
# ===========================================================================
print("\n" + "="*60)
print("=== 7A — TARGET OUTLIERS ===")
print("="*60)

for target_col in ['TARGET_3Y', 'TARGET_5Y']:
    nonzero = agg_2021[agg_2021[target_col] > 0][target_col]
    pcts = np.percentile(nonzero, [90, 95, 99, 99.5, 99.9])
    print(f"\n{target_col} non-zero ({len(nonzero):,} clienti):")
    for label, val in zip(['p90','p95','p99','p99.5','p99.9'], pcts):
        print(f"  {label}: {val:,.0f} EUR")
    print(f"  max: {nonzero.max():,.0f} EUR")

# p99.9 threshold per TARGET_3Y
p999_3y = float(np.percentile(agg_2021[agg_2021['TARGET_3Y'] > 0]['TARGET_3Y'], 99.9))
top_999 = agg_2021[agg_2021['TARGET_3Y'] > p999_3y]
print(f"\nClienti oltre TARGET_3Y p99.9 (> {p999_3y:,.0f} EUR): {len(top_999):,}")
print(f"  TARGET_3Y medio: {top_999['TARGET_3Y'].mean():,.0f} EUR")
print(f"  TARGET_3Y max:   {top_999['TARGET_3Y'].max():,.0f} EUR")

# Top 50 per TARGET_3Y
cols_top50 = [c for c in ['TARGET_3Y','TARGET_5Y','TO_FULL_HIST','NB_TRS_FULL_HIST',
                           'RECENCY','SENIORITY','RESIDENCY_MARKET','GENDER'] if c in agg_2021.columns]
top50 = agg_2021.nlargest(50, 'TARGET_3Y')[cols_top50].copy().reset_index(drop=True)
top50.insert(0, 'Client_ID', [f'Client_{i+1:03d}' for i in range(len(top50))])
top50.to_csv(OUT_TABLES / 'target_outliers_top50.csv', index=False)
print(f"\nTop 50 per TARGET_3Y — statistiche:")
for col in ['TO_FULL_HIST','NB_TRS_FULL_HIST','SENIORITY','RECENCY']:
    if col in top50.columns:
        print(f"  {col}: mean={top50[col].mean():.1f}, min={top50[col].min():.1f}, max={top50[col].max():.1f}")
print("  RESIDENCY_MARKET top 5:", top50['RESIDENCY_MARKET'].value_counts().head(5).to_dict() if 'RESIDENCY_MARKET' in top50.columns else "N/A")
print("  GENDER:", top50['GENDER'].value_counts().to_dict() if 'GENDER' in top50.columns else "N/A")

# Verifica TARGET_3Y <= TARGET_5Y
both_nonull = agg_2021[['TARGET_3Y','TARGET_5Y']].dropna()
violations_target = (both_nonull['TARGET_3Y'] > both_nonull['TARGET_5Y']).sum()
print(f"\nViolazioni TARGET_3Y > TARGET_5Y: {violations_target:,} ({violations_target/len(both_nonull)*100:.3f}%)")
if violations_target > 0:
    viol_df = agg_2021[agg_2021['TARGET_3Y'] > agg_2021['TARGET_5Y']][['TARGET_3Y','TARGET_5Y']].head(20)
    viol_df['diff'] = viol_df['TARGET_3Y'] - viol_df['TARGET_5Y']
    print("  Distribuzione diff (TARGET_3Y - TARGET_5Y):")
    print(f"  min={viol_df['diff'].min():.2f}, median={viol_df['diff'].median():.2f}, max={viol_df['diff'].max():.2f}")

print_finding("TARGET OUTLIERS",
    f"p99.9 TARGET_3Y: {p999_3y:,.0f} EUR. Clienti oltre: {len(top_999):,}. "
    f"Violazioni TARGET_3Y > TARGET_5Y: {violations_target:,}. "
    "Top 50 salvato: target_outliers_top50.csv."
)


# ===========================================================================
# 7B — Spend storico outliers
# ===========================================================================
print("\n" + "="*60)
print("=== 7B — SPEND STORICO OUTLIERS ===")
print("="*60)

for col in ['TO_FULL_HIST', 'TO_PAST_3Y']:
    if col not in agg_2021.columns:
        continue
    vals = agg_2021[col].dropna()
    p99 = float(np.percentile(vals, 99))
    p999 = float(np.percentile(vals, 99.9))
    print(f"\n{col}: p99={p99:,.0f}, p99.9={p999:,.0f}, max={vals.max():,.0f}")
    top_p999 = agg_2021[agg_2021[col] > p999]
    print(f"  Clienti > p99.9: {len(top_p999):,}")

# Correlazione TO_FULL_HIST vs TARGET_3Y nel segmento top p99.9
p999_hist = float(np.percentile(agg_2021['TO_FULL_HIST'].dropna(), 99.9))
top_hist   = set(agg_2021[agg_2021['TO_FULL_HIST'] > p999_hist]['CLIENT_ID'])
top_target = set(agg_2021[agg_2021['TARGET_3Y'] > p999_3y]['CLIENT_ID'])
overlap_ht = len(top_hist & top_target)
print(f"\nOverlap top TO_FULL_HIST (p99.9) e top TARGET_3Y (p99.9): {overlap_ht:,} clienti")
print(f"  % di top_hist anche in top_target: {overlap_ht/len(top_hist)*100:.1f}%")

# Top 10 per TO_FULL_HIST — sanity check narrativo
top10_hist = agg_2021.nlargest(10, 'TO_FULL_HIST')
print("\nTop 10 per TO_FULL_HIST — sanity check:")
for _, row in top10_hist.iterrows():
    nb = row.get('NB_TRS_FULL_HIST', float('nan'))
    sen = row.get('SENIORITY', float('nan'))
    rec = row.get('RECENCY', float('nan'))
    t3y = row.get('TARGET_3Y', float('nan'))
    print(f"  TO_FULL_HIST={row['TO_FULL_HIST']:>12,.0f} | NB_TRS={nb:>4.0f} | SENIORITY={sen:>5.0f} | RECENCY={rec:>5.0f} | TARGET_3Y={t3y:>10,.0f}")

# Clienti con TO_FULL_HIST=0 ma TARGET_3Y>0
zero_hist_pos_target = agg_2021[(agg_2021['TO_FULL_HIST'] == 0) & (agg_2021['TARGET_3Y'] > 0)]
print(f"\nTO_FULL_HIST=0 con TARGET_3Y>0: {len(zero_hist_pos_target):,}")
if len(zero_hist_pos_target) > 0:
    print(f"  TARGET_3Y medio: {zero_hist_pos_target['TARGET_3Y'].mean():,.0f}")
    print(f"  SENIORITY medio: {zero_hist_pos_target['SENIORITY'].mean():.1f}")

print_finding("SPEND STORICO OUTLIERS",
    f"TO_FULL_HIST p99.9: {p999_hist:,.0f} EUR. "
    f"Overlap top TO_FULL_HIST e top TARGET_3Y: {overlap_ht} clienti ({overlap_ht/len(top_hist)*100:.0f}%). "
    f"TO_FULL_HIST=0 con TARGET_3Y>0: {len(zero_hist_pos_target):,}."
)


# ===========================================================================
# 7C — Frequency outliers (NB_TRS, QTY_PDT)
# ===========================================================================
print("\n" + "="*60)
print("=== 7C — FREQUENCY OUTLIERS ===")
print("="*60)

# NB_TRS in Aggregated
nb_col = 'NB_TRS_FULL_HIST'
if nb_col in agg_2021.columns:
    nb_vals = agg_2021[nb_col].dropna()
    pcts_nb = np.percentile(nb_vals, [90, 95, 99, 99.9])
    print(f"\n{nb_col}: p90={pcts_nb[0]:.0f}, p95={pcts_nb[1]:.0f}, p99={pcts_nb[2]:.0f}, "
          f"p99.9={pcts_nb[3]:.0f}, max={nb_vals.max():.0f}")
    p999_nb = float(pcts_nb[3])
    top_nb = agg_2021[agg_2021[nb_col] > p999_nb]
    print(f"Clienti NB_TRS > p99.9 ({p999_nb:.0f}): {len(top_nb):,}")
    print(f"  TO_FULL_HIST medio: {top_nb['TO_FULL_HIST'].mean():,.0f}")
    print(f"  TO_FULL_HIST mediana: {top_nb['TO_FULL_HIST'].median():,.0f}")

# QTY_PDT in Transactions
print(f"\nQTY_PDT in Transactions:")
qty_vals = trs['QTY_PDT'].dropna()
pcts_qty = np.percentile(qty_vals, [90, 95, 99, 99.9])
print(f"  p90={pcts_qty[0]:.0f}, p95={pcts_qty[1]:.0f}, p99={pcts_qty[2]:.0f}, "
      f"p99.9={pcts_qty[3]:.1f}, max={qty_vals.max():.0f}")
high_qty = trs[trs['QTY_PDT'] > 10]
print(f"Transazioni con QTY_PDT > 10: {len(high_qty):,}")
if len(high_qty) > 0:
    print("  CHANNEL:", high_qty['CHANNEL'].value_counts().to_dict())
    print("  TRS_CATEG:", high_qty['TRS_CATEG'].value_counts().to_dict())
    print("  CATEG:", high_qty['CATEG'].value_counts().to_dict())
    print(f"  TO_WITHOUTTAX medio: {high_qty['TO_WITHOUTTAX_EUR_CONST'].mean():,.0f}")

# NB_TRS alto ma TO_FULL_HIST basso (molte transazioni a basso valore)
if nb_col in agg_2021.columns:
    p99_nb_val = float(np.percentile(nb_vals, 99))
    p10_hist   = float(np.percentile(agg_2021['TO_FULL_HIST'].dropna(), 10))
    high_nb_low_hist = agg_2021[(agg_2021[nb_col] > p99_nb_val) &
                                (agg_2021['TO_FULL_HIST'] < p10_hist)]
    print(f"\nNB_TRS > p99 ({p99_nb_val:.0f}) MA TO_FULL_HIST < p10 ({p10_hist:.0f}): {len(high_nb_low_hist):,}")

# Salva tabella frequency
freq_rows = []
if nb_col in agg_2021.columns:
    for threshold in [5, 10, 20, 50]:
        n_above = int((agg_2021[nb_col] > threshold).sum())
        freq_rows.append({'Colonna': nb_col, 'Soglia': threshold, 'N sopra soglia': n_above,
                          '% totale': round(n_above/N*100, 2)})
for threshold in [5, 10, 20]:
    n_above = int((trs['QTY_PDT'] > threshold).sum())
    freq_rows.append({'Colonna': 'QTY_PDT (TRS)', 'Soglia': threshold, 'N sopra soglia': n_above,
                      '% totale': round(n_above/len(trs)*100, 3)})
pd.DataFrame(freq_rows).to_csv(OUT_TABLES / 'frequency_outliers.csv', index=False)

print_finding("FREQUENCY OUTLIERS",
    f"NB_TRS p99.9: {p999_nb:.0f}. Clienti > p99.9: {len(top_nb):,}. "
    f"QTY_PDT max: {qty_vals.max():.0f}. Transazioni QTY_PDT>10: {len(high_qty):,}. "
    "Salvato: frequency_outliers.csv."
)


# ===========================================================================
# 7D — Price outliers
# ===========================================================================
print("\n" + "="*60)
print("=== 7D — PRICE OUTLIERS ===")
print("="*60)

# ARTICLE_WWPRICE in TRS
wwprice = trs['ARTICLE_WWPRICE'].dropna()
pcts_ww = np.percentile(wwprice[wwprice > 0], [99, 99.9])
print(f"\nARTICLE_WWPRICE (non-zero): p99={pcts_ww[0]:,.0f}, p99.9={pcts_ww[1]:,.0f}, max={wwprice.max():,.0f}")
top_ww = trs[trs['ARTICLE_WWPRICE'] == wwprice.max()]
print(f"ARTICLE_ID con WWPRICE max ({wwprice.max():,.0f} EUR): {top_ww['ARTICLE_ID'].unique()}")
print(f"  CATEG: {top_ww['CATEG'].unique()}, CHANNEL: {top_ww['CHANNEL'].unique()}")

# WORLD_PRICE in Articles
wp = art['WORLD_PRICE'].dropna()
print(f"\nWORLD_PRICE Articles: min={wp.min():.2f}, p99={np.percentile(wp,99):,.0f}, "
      f"p99.9={np.percentile(wp,99.9):,.0f}, max={wp.max():,.0f}")
n_100k = (wp > 100_000).sum()
print(f"Articoli WORLD_PRICE > 100.000 EUR: {n_100k:,} ({n_100k/len(wp)*100:.1f}%)")
art_high = art[art['WORLD_PRICE'] > 100_000]
print("  PRODUCT_CATEGORY:", art_high['PRODUCT_CATEGORY'].value_counts().to_dict())
for flag in ['FLAG_HE','FLAG_BRIDAL','FLAG_DIAMOND']:
    if flag in art_high.columns:
        print(f"  {flag}: {art_high[flag].value_counts().to_dict()}")

# WORLD_PRICE < 1 EUR
n_lt1 = (wp < 1).sum()
print(f"\nArticoli WORLD_PRICE < 1 EUR: {n_lt1:,}")
if n_lt1 > 0:
    print("  PRODUCT_CATEGORY:", art[art['WORLD_PRICE'] < 1]['PRODUCT_CATEGORY'].value_counts().to_dict())

# CASO APERTO: WWPRICE=0 e TO_WITHOUTTAX>0
zero_ww_pos_to = trs[(trs['ARTICLE_WWPRICE'] == 0) & (trs['TO_WITHOUTTAX_EUR_CONST'] > 0)]
print(f"\n=== CASO APERTO: WWPRICE=0 con TO_WITHOUTTAX>0 ===")
print(f"N righe: {len(zero_ww_pos_to):,}")
print(f"  TRS_CATEG: {zero_ww_pos_to['TRS_CATEG'].value_counts().to_dict()}")
print(f"  CHANNEL: {zero_ww_pos_to['CHANNEL'].value_counts().to_dict()}")
print(f"  CATEG: {zero_ww_pos_to['CATEG'].value_counts().to_dict()}")
to_dist = zero_ww_pos_to['TO_WITHOUTTAX_EUR_CONST']
print(f"  TO_WITHOUTTAX: mean={to_dist.mean():,.0f}, median={to_dist.median():,.0f}, "
      f"p99={np.percentile(to_dist,99):,.0f}, max={to_dist.max():,.0f}")
# Anno di acquisto
print(f"  Anno TRS_DATE: {zero_ww_pos_to['TRS_DATE'].dt.year.value_counts().sort_index().to_dict()}")

# Classificazione: repair con costo di servizio, non articolo prezzato
repair_wwzero = zero_ww_pos_to[zero_ww_pos_to['TRS_CATEG'] == 'Repair']
sale_wwzero   = zero_ww_pos_to[zero_ww_pos_to['TRS_CATEG'] == 'Sale']
print(f"\n  Repair (costo servizio senza articolo prezzato): {len(repair_wwzero):,}")
print(f"  Sale (articolo non catalogato con prezzo): {len(sale_wwzero):,}")

# Salva price outliers
price_rows = [
    {'Check': 'WWPRICE max in TRS', 'Value': wwprice.max(), 'ARTICLE_ID': str(top_ww['ARTICLE_ID'].unique()), 'Classificazione': 'LEGITTIMO - alta gioielleria'},
    {'Check': 'WORLD_PRICE max in Articles', 'Value': wp.max(), 'ARTICLE_ID': '', 'Classificazione': 'LEGITTIMO - alta gioielleria'},
    {'Check': 'Articoli WORLD_PRICE > 100k', 'Value': n_100k, 'ARTICLE_ID': '', 'Classificazione': 'LEGITTIMO'},
    {'Check': 'Articoli WORLD_PRICE < 1 EUR', 'Value': n_lt1, 'ARTICLE_ID': '', 'Classificazione': 'INCERTO'},
    {'Check': 'WWPRICE=0 con TO>0 (Repair)', 'Value': len(repair_wwzero), 'ARTICLE_ID': '', 'Classificazione': 'LEGITTIMO - costo servizio riparazione'},
    {'Check': 'WWPRICE=0 con TO>0 (Sale)', 'Value': len(sale_wwzero), 'ARTICLE_ID': '', 'Classificazione': 'INCERTO - articoli non catalogati'},
]
pd.DataFrame(price_rows).to_csv(OUT_TABLES / 'price_outliers.csv', index=False)

print_finding("PRICE OUTLIERS",
    f"WWPRICE max: {wwprice.max():,.0f} EUR. WORLD_PRICE max: {wp.max():,.0f} EUR. "
    f"WWPRICE=0 con TO>0: {len(zero_ww_pos_to):,} ({len(repair_wwzero):,} Repair + {len(sale_wwzero):,} Sale). "
    "Classificazione: Repair=LEGITTIMO (costo servizio), Sale=INCERTO. "
    "Salvato: price_outliers.csv."
)


# ===========================================================================
# 7E — Outliers demografici
# ===========================================================================
print("\n" + "="*60)
print("=== 7E — DEMOGRAPHIC OUTLIERS ===")
print("="*60)

# Parse date
cli_work = cli.copy()
for col in ['BIRTH_DATE', 'FIRST_PURCHASE_DATE', 'FIRST_TRANSACTION_DATE']:
    if col in cli_work.columns:
        cli_work[col] = pd.to_datetime(cli_work[col], errors='coerce')

ref_date = pd.Timestamp('2021-01-01')

# BIRTH_DATE anomale
bd = cli_work['BIRTH_DATE'].dropna()
print(f"\nBIRTH_DATE non-null: {len(bd):,} ({len(bd)/len(cli_work)*100:.1f}%)")
bd_pre1900 = (bd.dt.year < 1900).sum()
print(f"BIRTH_DATE prima del 1900: {bd_pre1900:,}")
print(f"  Top 5 date pre-1900: {bd[bd.dt.year < 1900].value_counts().head(5).to_dict()}")

# Eta al 2021-01-01
cli_work['age_at_2021'] = (ref_date - cli_work['BIRTH_DATE']).dt.days / 365.25
too_young = (cli_work['age_at_2021'] < 18).sum()
too_old   = (cli_work['age_at_2021'] > 100).sum()
print(f"\nEta calcolata al 2021-01-01:")
print(f"  Eta < 18 anni: {too_young:,}")
print(f"  Eta > 100 anni: {too_old:,}")
print(f"  Plausibili (18-100): {((cli_work['age_at_2021'] >= 18) & (cli_work['age_at_2021'] <= 100)).sum():,}")

# FIRST_PURCHASE_DATE vs FIRST_TRANSACTION_DATE
if 'FIRST_PURCHASE_DATE' in cli_work.columns and 'FIRST_TRANSACTION_DATE' in cli_work.columns:
    both = cli_work[cli_work['FIRST_PURCHASE_DATE'].notna() & cli_work['FIRST_TRANSACTION_DATE'].notna()].copy()
    both['date_diff'] = (both['FIRST_PURCHASE_DATE'] - both['FIRST_TRANSACTION_DATE']).dt.days.abs()
    gt365 = (both['date_diff'] > 365).sum()
    print(f"\nFIRST_PURCHASE_DATE vs FIRST_TRANSACTION_DATE:")
    print(f"  Differenza > 365 giorni: {gt365:,} ({gt365/len(both)*100:.1f}%)")
    print(f"  Max differenza: {both['date_diff'].max():.0f} giorni")
    print(f"  Mediana differenza: {both['date_diff'].median():.0f} giorni")

# SENIORITY = 0 in Aggregated_Data snapshot 2021
if 'SENIORITY' in agg_2021.columns:
    sen0 = (agg_2021['SENIORITY'] == 0).sum()
    print(f"\nSENIORITY = 0 in snapshot 2021: {sen0:,} ({sen0/N*100:.2f}%)")
    neg_sen = (agg_2021['SENIORITY'] < 0).sum()
    print(f"SENIORITY < 0: {neg_sen:,}")
    print(f"SENIORITY: min={agg_2021['SENIORITY'].min():.0f}, max={agg_2021['SENIORITY'].max():.0f}, "
          f"mean={agg_2021['SENIORITY'].mean():.0f}")

# Salva
demo_rows = [
    {'Check': 'BIRTH_DATE pre-1900', 'N': int(bd_pre1900), 'Classificazione': 'ERRORE - placeholder impossibile'},
    {'Check': 'Eta < 18 anni al 2021', 'N': int(too_young), 'Classificazione': 'ERRORE - impossibile o placeholder'},
    {'Check': 'Eta > 100 anni al 2021', 'N': int(too_old), 'Classificazione': 'ERRORE - implausibile'},
    {'Check': 'FIRST_PURCHASE vs FIRST_TRS > 365gg', 'N': int(gt365) if 'gt365' in dir() else -1, 'Classificazione': 'INCERTO'},
    {'Check': 'SENIORITY = 0 (snapshot 2021)', 'N': int(sen0) if 'SENIORITY' in agg_2021.columns else -1, 'Classificazione': 'LEGITTIMO - cliente nuovo'},
]
pd.DataFrame(demo_rows).to_csv(OUT_TABLES / 'demographic_outliers.csv', index=False)

print_finding("DEMOGRAPHIC OUTLIERS",
    f"BIRTH_DATE pre-1900: {bd_pre1900:,}. Eta<18: {too_young:,}. Eta>100: {too_old:,}. "
    f"SENIORITY=0: {sen0:,}. Salvato: demographic_outliers.csv."
)


# ===========================================================================
# 7F — CRC e CCP outliers
# ===========================================================================
print("\n" + "="*60)
print("=== 7F — CRC/CCP OUTLIERS ===")
print("="*60)

# CRC APPOINTMENT_DURATION
dur_col = 'APPOINTMENT_DURATION'
if dur_col in crc.columns:
    dur_nonull = crc[dur_col].dropna()
    pcts_dur = np.percentile(dur_nonull, [95, 99])
    print(f"\nAPPOINTMENT_DURATION (non-null {len(dur_nonull):,} valori):")
    print(f"  p95={pcts_dur[0]:,.0f}, p99={pcts_dur[1]:,.0f}, max={dur_nonull.max():,.0f}")
    anomali_dur = dur_nonull[dur_nonull > 480]  # > 8 ore in minuti
    print(f"  Durata > 480 min (8 ore): {len(anomali_dur):,}")
    if len(anomali_dur) > 0:
        print(f"  Valori anomali: {sorted(anomali_dur.unique())[:10]}")

# Top 10 clienti per N interazioni CRC
top_crc = crc.groupby('CLIENT_ID').size().nlargest(10)
print(f"\nTop 10 clienti per N interazioni CRC:")
print(top_crc.to_string())

# CCP — 228 violazioni SALE_DATE > CREATION_DATE
valid = ccp[ccp['SALE_DATE'].notna() & ccp['CREATION_DATE'].notna()].copy()
valid['gap'] = (valid['SALE_DATE'] - valid['CREATION_DATE']).dt.days
violations_ccp = valid[valid['gap'] > 0]
print(f"\nCCP violazioni SALE_DATE > CREATION_DATE ({len(violations_ccp):,}):")
if len(violations_ccp) > 0:
    print(f"  min={violations_ccp['gap'].min():.0f}gg, median={violations_ccp['gap'].median():.0f}gg, max={violations_ccp['gap'].max():.0f}gg")
    print(f"  Violazioni <= 7 giorni: {(violations_ccp['gap'] <= 7).sum():,} (arrotondamento)")
    print(f"  Violazioni > 30 giorni: {(violations_ccp['gap'] > 30).sum():,} (errore serio)")

# Top 10 clienti per N prodotti CCP
top_ccp = ccp.groupby('CLIENT_ID').size().nlargest(10)
print(f"\nTop 10 clienti per N prodotti CCP:")
print(top_ccp.to_string())

print_finding("CRC CCP OUTLIERS",
    f"APPOINTMENT_DURATION max: {dur_nonull.max():,.0f}. > 8 ore: {len(anomali_dur):,}. "
    f"CCP violazioni SALE>CREATION: {len(violations_ccp):,}. "
    f"Di cui <= 7gg (prob. arrotondamento): {(violations_ccp['gap'] <= 7).sum():,}."
)


# ===========================================================================
# 7G — Sintesi classificazione outliers
# ===========================================================================
print("\n" + "="*60)
print("=== 7G — OUTLIER SUMMARY ===")
print("="*60)

p999_3y_val = float(np.percentile(agg_2021[agg_2021['TARGET_3Y'] > 0]['TARGET_3Y'], 99.9))

summary = [
    {'Dataset': 'Aggregated_2021', 'Colonna': 'TARGET_3Y',
     'Tipo outlier': 'Valori estremi coda destra (> p99.9)',
     'N record': int(len(agg_2021[agg_2021['TARGET_3Y'] > p999_3y_val])),
     '% totale': round(len(agg_2021[agg_2021['TARGET_3Y'] > p999_3y_val])/N*100, 3),
     'Classificazione': 'LEGITTIMO',
     'Ipotesi causa': 'VIC (Very Important Clients) - comportamento reale atteso'},
    {'Dataset': 'Aggregated_2021', 'Colonna': 'TARGET_3Y > TARGET_5Y',
     'Tipo outlier': 'Violazione logica',
     'N record': int(violations_target),
     '% totale': round(violations_target/len(both_nonull)*100, 3),
     'Classificazione': 'ERRORE' if violations_target > 0 else 'OK',
     'Ipotesi causa': 'Anomalia nella costruzione del target'},
    {'Dataset': 'Aggregated_2021', 'Colonna': 'TO_FULL_HIST',
     'Tipo outlier': 'Spend storico estremo (> p99.9)',
     'N record': int(len(agg_2021[agg_2021['TO_FULL_HIST'] > p999_hist])),
     '% totale': round(len(agg_2021[agg_2021['TO_FULL_HIST'] > p999_hist])/N*100, 3),
     'Classificazione': 'LEGITTIMO',
     'Ipotesi causa': 'Clienti storici VIC con decenni di acquisti'},
    {'Dataset': 'Aggregated_2021', 'Colonna': 'TO_FULL_HIST=0 con TARGET>0',
     'Tipo outlier': 'Contraddizione logica',
     'N record': int(len(zero_hist_pos_target)),
     '% totale': round(len(zero_hist_pos_target)/N*100, 3),
     'Classificazione': 'INCERTO',
     'Ipotesi causa': 'Clienti nuovi o storico pre-dataset'},
    {'Dataset': 'Transactions', 'Colonna': 'QTY_PDT',
     'Tipo outlier': f'QTY_PDT > 10 ({qty_vals.max():.0f} max)',
     'N record': int(len(high_qty)),
     '% totale': round(len(high_qty)/len(trs)*100, 3),
     'Classificazione': 'INCERTO',
     'Ipotesi causa': 'Acquisti multipli/wholesale o anomalia di inserimento'},
    {'Dataset': 'Transactions', 'Colonna': 'ARTICLE_WWPRICE=0, TO>0 (Repair)',
     'Tipo outlier': 'Prezzo articolo mancante su transazione con valore',
     'N record': int(len(repair_wwzero)),
     '% totale': round(len(repair_wwzero)/len(trs)*100, 2),
     'Classificazione': 'LEGITTIMO',
     'Ipotesi causa': 'Costo di servizio riparazione - articolo non prezzato nel catalogo'},
    {'Dataset': 'Transactions', 'Colonna': 'ARTICLE_WWPRICE=0, TO>0 (Sale)',
     'Tipo outlier': 'Prezzo articolo mancante su vendita con valore',
     'N record': int(len(sale_wwzero)),
     '% totale': round(len(sale_wwzero)/len(trs)*100, 2),
     'Classificazione': 'INCERTO',
     'Ipotesi causa': 'Articoli legacy/custom non catalogati - mantenere con flag WWPRICE_MISSING'},
    {'Dataset': 'Articles', 'Colonna': 'WORLD_PRICE',
     'Tipo outlier': f'WORLD_PRICE > 100k EUR ({n_100k:,} articoli)',
     'N record': int(n_100k),
     '% totale': round(n_100k/len(art)*100, 1),
     'Classificazione': 'LEGITTIMO',
     'Ipotesi causa': 'Alta gioielleria Cartier (diamanti, pezzi unici)'},
    {'Dataset': 'Articles', 'Colonna': 'WORLD_PRICE < 1 EUR',
     'Tipo outlier': 'Prezzo quasi-zero',
     'N record': int(n_lt1),
     '% totale': round(n_lt1/len(art)*100, 2),
     'Classificazione': 'INCERTO',
     'Ipotesi causa': 'Accessori minori, servizi o errori di catalogazione'},
    {'Dataset': 'Clients', 'Colonna': 'BIRTH_DATE',
     'Tipo outlier': 'Data nascita pre-1900 o eta<18/eta>100',
     'N record': int(bd_pre1900 + too_young),
     '% totale': round((bd_pre1900 + too_young)/len(cli)*100, 2),
     'Classificazione': 'ERRORE',
     'Ipotesi causa': 'Placeholder (1804-01-01) o dato mancante codificato'},
    {'Dataset': 'CRC', 'Colonna': 'APPOINTMENT_DURATION',
     'Tipo outlier': 'Durata > 8 ore',
     'N record': int(len(anomali_dur)),
     '% totale': round(len(anomali_dur)/len(crc)*100, 3),
     'Classificazione': 'ERRORE',
     'Ipotesi causa': 'Errore di inserimento durata appuntamento'},
    {'Dataset': 'CCP', 'Colonna': 'SALE_DATE > CREATION_DATE',
     'Tipo outlier': 'Violazione logica date',
     'N record': int(len(violations_ccp)),
     '% totale': round(len(violations_ccp)/len(ccp)*100, 2),
     'Classificazione': 'ERRORE',
     'Ipotesi causa': f"{(violations_ccp['gap']<=7).sum()} arrotondamento, {(violations_ccp['gap']>30).sum()} errori seri"},
]

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_TABLES / 'outlier_classification.csv', index=False)
print("\nTabella di classificazione outliers:")
print(summary_df[['Dataset','Colonna','N record','% totale','Classificazione']].to_string(index=False))

print_finding("OUTLIER SUMMARY",
    f"Totale pattern outlier analizzati: {len(summary_df)}. "
    f"LEGITTIMO: {(summary_df['Classificazione']=='LEGITTIMO').sum()}. "
    f"ERRORE: {(summary_df['Classificazione']=='ERRORE').sum()}. "
    f"INCERTO: {(summary_df['Classificazione']=='INCERTO').sum()}. "
    "Salvato: outlier_classification.csv."
)

print("\n=== TABELLE SALVATE ===")
for f in sorted(OUT_TABLES.iterdir()):
    print(f"  {f.name}")
print("\n=== ANALISI FASE 7 COMPLETATA ===")
