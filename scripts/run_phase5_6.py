"""
Script di verifica Fase 5+6 — Quality Analysis e Referential Integrity.
Eseguire dalla root del progetto: python3 scripts/run_phase5_6.py
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
crc = dfs['CRC']
ccp = dfs['CCP']
art = dfs['Articles']
cli = dfs['Clients']

agg_2021 = agg[agg['DATE_TARGET'].dt.year == 2021].copy()
clients_2021 = set(agg_2021['CLIENT_ID'].dropna().unique())
snap_2021 = pd.Timestamp('2021-01-01')
DATE_TARGETS = sorted(agg['DATE_TARGET'].dropna().unique())
snap_last = DATE_TARGETS[-1]
col_spend = 'TO_WITHOUTTAX_EUR_CONST'

print(f"\nSnapshot 2021: {len(clients_2021):,} CLIENT_ID unici")

# ===========================================================================
# FASE 5A — Copertura temporale Transactions
# ===========================================================================
rows_cov = []
for snap in DATE_TARGETS:
    safe  = int((trs['TRS_DATE'] <= snap).sum())
    leak  = int((trs['TRS_DATE'] >  snap).sum())
    total = safe + leak
    rows_cov.append({
        'DATE_TARGET':       str(snap)[:10],
        'TRS safe (<= DT)':  safe,
        'TRS leakage (> DT)': leak,
        'Total':             total,
        '% leakage':         round(leak / total * 100, 2),
    })
cov_df = pd.DataFrame(rows_cov)
cov_df.to_csv(OUT_TABLES / 'transactions_temporal_coverage.csv', index=False)
trs_leak_last = trs[trs['TRS_DATE'] > snap_last]
print("\n=== 5A ===")
print(cov_df.to_string(index=False))
print(f"Post-{str(snap_last)[:10]}: {len(trs_leak_last):,} ({len(trs_leak_last)/len(trs)*100:.1f}%)")
if len(trs_leak_last) > 0:
    print("CHANNEL post-ultimo-snapshot:", trs_leak_last['CHANNEL'].value_counts().to_dict())
print_finding("TRANSACTIONS COPERTURA TEMPORALE",
    f"TRS_DATE: {trs['TRS_DATE'].min()} -- {trs['TRS_DATE'].max()}. "
    f"Post-2021: {len(trs_leak_last):,} ({len(trs_leak_last)/len(trs)*100:.1f}%)."
)

# ===========================================================================
# FASE 5B — SERIAL_NUMBER
# ===========================================================================
PLACEHOLDER = trs['SERIAL_NUMBER'].value_counts().idxmax() if trs['SERIAL_NUMBER'].notna().any() else None
n_null = int(trs['SERIAL_NUMBER'].isna().sum())
n_ph   = int((trs['SERIAL_NUMBER'] == PLACEHOLDER).sum()) if PLACEHOLDER else 0
print("\n=== 5B ===")
print(f"NULL: {n_null:,} ({n_null/len(trs)*100:.1f}%)")
print(f"Placeholder: {PLACEHOLDER!r} - {n_ph:,} ({n_ph/len(trs)*100:.1f}%)")

sn_rows = []
for col in ['TRS_CATEG', 'CHANNEL', 'CATEG']:
    for val in trs[col].dropna().unique():
        mask_val  = trs[col] == val
        tot       = int(mask_val.sum())
        null_v    = int((mask_val & trs['SERIAL_NUMBER'].isna()).sum())
        ph_v      = int((mask_val & (trs['SERIAL_NUMBER'] == PLACEHOLDER)).sum()) if PLACEHOLDER else 0
        sn_rows.append({
            'Dimensione': col, 'Valore': val, 'N totale': tot,
            'N NULL SN': null_v, '% NULL SN': round(null_v/tot*100, 1),
            'N Placeholder SN': ph_v, '% Placeholder SN': round(ph_v/tot*100, 1),
        })
sn_df = pd.DataFrame(sn_rows)
sn_df.to_csv(OUT_TABLES / 'serial_number_analysis.csv', index=False)

print("Distribuzione per TRS_CATEG:")
print(trs[trs['SERIAL_NUMBER'].isna()]['TRS_CATEG'].value_counts().to_dict())
if PLACEHOLDER:
    print("Placeholder per CHANNEL:")
    print(trs[trs['SERIAL_NUMBER'] == PLACEHOLDER]['CHANNEL'].value_counts().to_dict())
    print("NULL per CHANNEL:")
    print(trs[trs['SERIAL_NUMBER'].isna()]['CHANNEL'].value_counts().to_dict())

print_finding("SERIAL_NUMBER",
    f"NULL: {n_null:,} ({n_null/len(trs)*100:.1f}%). "
    f"Placeholder '{PLACEHOLDER}': {n_ph:,} ({n_ph/len(trs)*100:.1f}%). "
    "Salvato: serial_number_analysis.csv."
)

# ===========================================================================
# FASE 5C — Valori negativi TO_WITHOUTTAX_EUR_CONST
# ===========================================================================
neg_mask  = trs[col_spend] < 0
null_mask = trs[col_spend].isna()
neg_trs   = trs[neg_mask]
neg_vals  = neg_trs[col_spend]

print("\n=== 5C ===")
print(f"NULL {col_spend}: {null_mask.sum():,}")
print(f"Valori < 0: {neg_mask.sum():,} ({neg_mask.mean()*100:.3f}%)")
print(f"Distribuzione: min={neg_vals.min():.2f}, p25={np.percentile(neg_vals,25):.2f}, "
      f"p50={np.percentile(neg_vals,50):.2f}, p75={np.percentile(neg_vals,75):.2f}, max={neg_vals.max():.2f}")
print("TRS_CATEG dei negativi:", neg_trs['TRS_CATEG'].value_counts().to_dict())
print("CHANNEL dei negativi:", neg_trs['CHANNEL'].value_counts().to_dict())
print("CATEG dei negativi:", neg_trs['CATEG'].value_counts().to_dict())

clients_with_neg = neg_trs['CLIENT_ID'].unique()
sample_clients   = clients_with_neg[:500]
trs_s = trs[trs['CLIENT_ID'].isin(sample_clients)]
matched = 0
for c in sample_clients:
    ct = trs_s[trs_s['CLIENT_ID'] == c]
    neg_a = set(ct[ct[col_spend] < 0]['ARTICLE_ID'].dropna())
    pos_a = set(ct[ct[col_spend] > 0]['ARTICLE_ID'].dropna())
    if neg_a & pos_a:
        matched += 1
print(f"Match ARTICLE_ID pos/neg stesso cliente: {matched}/{len(sample_clients)} ({matched/len(sample_clients)*100:.0f}%)")

pd.DataFrame([{
    'N<0': int(neg_mask.sum()), 'pct<0': round(neg_mask.mean()*100, 3),
    'N_NULL': int(null_mask.sum()),
    'min': round(neg_vals.min(), 2), 'p25': round(np.percentile(neg_vals, 25), 2),
    'p50': round(np.percentile(neg_vals, 50), 2), 'p75': round(np.percentile(neg_vals, 75), 2),
    'max': round(neg_vals.max(), 2),
    'N clienti con neg': len(clients_with_neg),
    'TRS_CATEG Sale': int((neg_trs['TRS_CATEG'] == 'Sale').sum()),
    'TRS_CATEG Repair': int((neg_trs['TRS_CATEG'] == 'Repair').sum()),
}]).to_csv(OUT_TABLES / 'quality_transactions_negatives.csv', index=False)

print_finding("VALORI NEGATIVI TRANSACTIONS",
    f"Righe < 0: {neg_mask.sum():,} ({neg_mask.mean()*100:.3f}%). "
    f"TRS_CATEG: {neg_trs['TRS_CATEG'].value_counts().to_dict()}. "
    f"Match articolo pos/neg (campione): {matched}/{len(sample_clients)}. "
    "Ipotesi: codificano resi (stessa categoria Sale). Salvato: quality_transactions_negatives.csv."
)

# ===========================================================================
# FASE 5D — ARTICLE_WWPRICE a zero
# ===========================================================================
zero_mask = trs['ARTICLE_WWPRICE'] == 0
zero_trs  = trs[zero_mask]

print("\n=== 5D ===")
print(f"ARTICLE_WWPRICE = 0: {zero_mask.sum():,} ({zero_mask.mean()*100:.2f}%)")
print("TRS_CATEG:", zero_trs['TRS_CATEG'].value_counts().to_dict())
print("CHANNEL:", zero_trs['CHANNEL'].value_counts().to_dict())
print(f"TO_WITHOUTTAX = 0 quando WWPRICE=0: {(zero_trs[col_spend] == 0).sum():,}")
print(f"TO_WITHOUTTAX > 0 quando WWPRICE=0: {(zero_trs[col_spend] > 0).sum():,}")
print(f"TO_WITHOUTTAX < 0 quando WWPRICE=0: {(zero_trs[col_spend] < 0).sum():,}")

print_finding("WWPRICE ZERO",
    f"Righe con ARTICLE_WWPRICE=0: {zero_mask.sum():,} ({zero_mask.mean()*100:.2f}%). "
    f"TRS_CATEG: {zero_trs['TRS_CATEG'].value_counts().to_dict()}. "
    "Nota: TO_WITHOUTTAX puo avere valore anche quando WWPRICE=0."
)

# ===========================================================================
# FASE 5E — Articles: mismatch categorie
# ===========================================================================
cats_articles = set(art['PRODUCT_CATEGORY'].dropna().unique())
cats_trs      = set(trs['CATEG'].dropna().unique())
cats_missing  = cats_articles - cats_trs
art_miss      = art[art['PRODUCT_CATEGORY'].isin(cats_missing)]
art_ids_mc    = set(art_miss['ARTICLE_ID'].dropna().unique())
overlap_e     = art_ids_mc & set(trs['ARTICLE_ID'].dropna().unique())

print("\n=== 5E ===")
print(f"Categorie Articles: {sorted(cats_articles)}")
print(f"Categorie Transactions: {sorted(cats_trs)}")
print(f"Mancanti: {sorted(cats_missing)}")
print(f"Articoli in categorie mancanti: {len(art_miss):,}")
print(f"ARTICLE_ID overlap con TRS: {len(overlap_e):,}")
if len(overlap_e) > 0:
    trs_ov = trs[trs['ARTICLE_ID'].isin(overlap_e)]
    print("CATEG in TRS per articoli 'mancanti':", trs_ov['CATEG'].value_counts().to_dict())
print("FLAG_HE per categorie mancanti:", art_miss['FLAG_HE'].value_counts().to_dict() if 'FLAG_HE' in art_miss.columns else "N/A")
print("FLAG_BRIDAL:", art_miss['FLAG_BRIDAL'].value_counts().to_dict() if 'FLAG_BRIDAL' in art_miss.columns else "N/A")

art.groupby('PRODUCT_CATEGORY').agg(
    n=('ARTICLE_ID', 'count'),
    price_med=('WORLD_PRICE', 'median'),
).assign(in_trs=lambda df: df.index.isin(cats_trs)).reset_index().to_csv(OUT_TABLES / 'articles_category_analysis.csv', index=False)

print_finding("CATEGORIE ARTICLES MANCANTI",
    f"Mancanti: {sorted(cats_missing)}. "
    f"Articoli: {len(art_miss):,}. "
    f"Overlap ARTICLE_ID con TRS: {len(overlap_e):,}. "
    "Ipotesi: macro-categorie TRS aggregano piu PRODUCT_CATEGORY Articles (mapping N:1)."
)

# ===========================================================================
# FASE 5F — CRC
# ===========================================================================
dur_col_list = [c for c in crc.columns if 'DURATION' in c.upper()]
dur_col = dur_col_list[0] if dur_col_list else None
crc_clients  = set(crc['CLIENT_ID'].dropna().unique())
crc_in_2021  = crc_clients & clients_2021

print("\n=== 5F ===")
print(f"CREATION_DATE: {crc['CREATION_DATE'].min()} -- {crc['CREATION_DATE'].max()}")
print(f"CLIENT_ID unici: {len(crc_clients):,}")
print(f"In snapshot 2021: {len(crc_in_2021):,} ({len(crc_in_2021)/len(clients_2021)*100:.1f}%)")
print(f"Duration col: {dur_col}")

if dur_col:
    print(f"Duration missing overall: {crc[dur_col].isna().mean()*100:.1f}%")
    origin_totals = crc.groupby('ORIGIN').size().rename('n_totale')
    dur_null_s    = crc[crc[dur_col].isna()].groupby('ORIGIN').size().rename('n_missing')
    dur_df = pd.concat([origin_totals, dur_null_s], axis=1).fillna(0).reset_index()
    dur_df['pct_missing'] = (dur_df['n_missing'] / dur_df['n_totale'] * 100).round(1)
    dur_df = dur_df.sort_values('pct_missing', ascending=False).reset_index(drop=True)
    print("\nMissing duration per ORIGIN:")
    print(dur_df.to_string(index=False))
    dur_df.to_csv(OUT_TABLES / 'crc_appointment_duration_by_origin.csv', index=False)
    std_val = dur_df['pct_missing'].std()
    diagnosi = "MAR - missing concentrato su certi ORIGIN" if std_val > 10 else "MNAR - distribuzione uniforme per ORIGIN"
    print(f"\nStd % missing per ORIGIN: {std_val:.1f}% => {diagnosi}")
else:
    diagnosi = "Colonna duration non trovata"

print_finding("CRC COPERTURA E APPOINTMENT_DURATION",
    f"Coverage snapshot 2021: {len(crc_in_2021):,} ({len(crc_in_2021)/len(clients_2021)*100:.1f}%). "
    f"Diagnosi duration: {diagnosi}. Salvato: crc_appointment_duration_by_origin.csv."
)

# ===========================================================================
# FASE 5G — CCP
# ===========================================================================
ccp_clients = set(ccp['CLIENT_ID'].dropna().unique())
ccp_in_2021 = ccp_clients & clients_2021
valid_mask  = ccp['SALE_DATE'].notna() & ccp['CREATION_DATE'].notna()
violations  = int((ccp.loc[valid_mask, 'SALE_DATE'] > ccp.loc[valid_mask, 'CREATION_DATE']).sum())

print("\n=== 5G ===")
print(f"CREATION_DATE: {ccp['CREATION_DATE'].min()} -- {ccp['CREATION_DATE'].max()}")
print(f"SALE_DATE: {ccp['SALE_DATE'].min()} -- {ccp['SALE_DATE'].max()}")
print(f"CLIENT_ID unici: {len(ccp_clients):,}")
print(f"In snapshot 2021: {len(ccp_in_2021):,} ({len(ccp_in_2021)/len(clients_2021)*100:.1f}%)")
print(f"Violazioni SALE_DATE > CREATION_DATE: {violations:,}")
if valid_mask.sum() > 0:
    gap = (ccp.loc[valid_mask, 'CREATION_DATE'] - ccp.loc[valid_mask, 'SALE_DATE']).dt.days
    print(f"Gap giorni: min={gap.min()}, median={gap.median():.0f}, max={gap.max()}")
print("% missing SALE_DATE per FLAG_GIFT:")
sale_miss = ccp.groupby('FLAG_GIFT')['SALE_DATE'].apply(lambda x: x.isna().mean()*100).round(1)
print(sale_miss.to_dict())

pd.DataFrame([{
    'N righe': len(ccp), 'N CLIENT_ID unici': len(ccp_clients),
    'N clienti snapshot 2021': len(ccp_in_2021),
    '% clienti 2021': round(len(ccp_in_2021)/len(clients_2021)*100, 1),
    'CREATION min': str(ccp['CREATION_DATE'].min()), 'CREATION max': str(ccp['CREATION_DATE'].max()),
    'SALE min': str(ccp['SALE_DATE'].min()), 'SALE max': str(ccp['SALE_DATE'].max()),
    'N violazioni SALE>CREATION': violations,
}]).to_csv(OUT_TABLES / 'quality_ccp.csv', index=False)

print_finding("CCP COPERTURA E DATE CONSISTENCY",
    f"Coverage 2021: {len(ccp_in_2021):,} ({len(ccp_in_2021)/len(clients_2021)*100:.1f}%). "
    f"Violazioni SALE>CREATION: {violations:,}. Salvato: quality_ccp.csv."
)

# ===========================================================================
# FASE 6A — Matrice Referential Integrity
# ===========================================================================
id_sets = {
    'Aggregated_2021': clients_2021,
    'Aggregated_ALL':  set(agg['CLIENT_ID'].dropna().unique()),
    'Transactions':    set(trs['CLIENT_ID'].dropna().unique()),
    'Clients':         set(cli['CLIENT_ID'].dropna().unique()),
    'CRC':             crc_clients,
    'CCP':             ccp_clients,
}
names = list(id_sets.keys())
rows  = []
for i, na in enumerate(names):
    for j, nb in enumerate(names):
        if i >= j:
            continue
        sa = id_sets[na]; sb = id_sets[nb]
        inter = sa & sb; oa = sa - sb; ob = sb - sa
        rows.append({
            'Dataset_A': na, 'Dataset_B': nb,
            'N_A': len(sa), 'N_B': len(sb),
            'N_intersection': len(inter),
            'N_only_A': len(oa), 'N_only_B': len(ob),
            'pct_A_in_B': round(len(inter)/len(sa)*100, 1) if sa else 0,
            'pct_B_in_A': round(len(inter)/len(sb)*100, 1) if sb else 0,
        })
ri_df = pd.DataFrame(rows)
ri_df.to_csv(OUT_TABLES / 'referential_integrity_matrix.csv', index=False)
print("\n=== 6A ===")
print(ri_df[['Dataset_A', 'Dataset_B', 'N_intersection', 'pct_A_in_B', 'pct_B_in_A']].to_string(index=False))

# ===========================================================================
# FASE 6B — Clienti orfani
# ===========================================================================
trs_not_agg2021  = id_sets['Transactions'] - clients_2021
agg2021_not_trs  = clients_2021 - id_sets['Transactions']
crc_not_agg2021  = crc_clients - clients_2021
ccp_not_agg2021  = ccp_clients - clients_2021

print("\n=== 6B ===")
print(f"TRS not in Agg2021:  {len(trs_not_agg2021):,}")
trs_orphan = trs[trs['CLIENT_ID'].isin(trs_not_agg2021)]
print(f"  Anno min/max transazioni orfane: {trs_orphan['TRS_DATE'].dt.year.min()}/{trs_orphan['TRS_DATE'].dt.year.max()}")
print(f"  Presenti in altri snapshot Agg: {len(trs_not_agg2021 & id_sets['Aggregated_ALL']):,}")
print(f"Agg2021 not in TRS:  {len(agg2021_not_trs):,}")
if 'TO_FULL_HIST' in agg_2021.columns:
    agg_orp = agg_2021[agg_2021['CLIENT_ID'].isin(agg2021_not_trs)]
    print(f"  Con TO_FULL_HIST>0 nonostante assenza in TRS: {(agg_orp['TO_FULL_HIST']>0).sum():,}")
print(f"CRC not in Agg2021:  {len(crc_not_agg2021):,}")
print(f"CCP not in Agg2021:  {len(ccp_not_agg2021):,}")

print_finding("CLIENTI ORFANI",
    f"TRS not in Agg2021: {len(trs_not_agg2021):,}. "
    f"Agg2021 not in TRS: {len(agg2021_not_trs):,}. "
    f"CRC not in Agg2021: {len(crc_not_agg2021):,}. "
    f"CCP not in Agg2021: {len(ccp_not_agg2021):,}."
)

# ===========================================================================
# FASE 6C — ARTICLE_ID coverage
# ===========================================================================
art_in_art = set(art['ARTICLE_ID'].dropna().unique())
art_in_trs = set(trs['ARTICLE_ID'].dropna().unique())
match_c    = art_in_trs & art_in_art
orphan_c   = art_in_trs - art_in_art

print("\n=== 6C ===")
print(f"ARTICLE_ID in TRS:           {len(art_in_trs):,}")
print(f"ARTICLE_ID in Articles:      {len(art_in_art):,}")
print(f"Match (TRS in Articles):     {len(match_c):,} ({len(match_c)/len(art_in_trs)*100:.1f}%)")
print(f"Orfani (TRS non in Articles):{len(orphan_c):,} ({len(orphan_c)/len(art_in_trs)*100:.1f}%)")
if len(orphan_c) > 0:
    trs_orp = trs[trs['ARTICLE_ID'].isin(orphan_c)]
    print("  CHANNEL orfani:", trs_orp['CHANNEL'].value_counts().to_dict())
    print("  CATEG orfani:", trs_orp['CATEG'].value_counts().to_dict())
    print(f"  Anno orfani: {trs_orp['TRS_DATE'].dt.year.min()}/{trs_orp['TRS_DATE'].dt.year.max()}")

pd.DataFrame([{
    'TRS ARTICLE_ID unici': len(art_in_trs),
    'Articles ARTICLE_ID unici': len(art_in_art),
    'Match': len(match_c), 'pct TRS in Art': round(len(match_c)/len(art_in_trs)*100, 1),
    'Orfani TRS': len(orphan_c),
}]).to_csv(OUT_TABLES / 'article_id_coverage.csv', index=False)

print_finding("ARTICLE_ID COVERAGE",
    f"TRS in Articles: {len(match_c):,}/{len(art_in_trs):,} ({len(match_c)/len(art_in_trs)*100:.1f}%). "
    f"Orfani: {len(orphan_c):,}. Salvato: article_id_coverage.csv."
)

# ===========================================================================
# FASE 6D — Sanity check
# ===========================================================================
trs_c    = trs[(trs['CLIENT_ID'].isin(clients_2021)) & (trs['TRS_DATE'] <= snap_2021)]
sum_trs  = trs_c[col_spend].sum()
sum_agg  = agg_2021['TO_FULL_HIST'].sum() if 'TO_FULL_HIST' in agg_2021.columns else 0
ratio    = sum_trs / sum_agg if sum_agg else 0
mean_trs = trs_c.groupby('CLIENT_ID')[col_spend].sum().mean()
mean_agg = agg_2021['TO_FULL_HIST'].mean() if 'TO_FULL_HIST' in agg_2021.columns else 0
mean_n_trs = trs_c.groupby('CLIENT_ID').size().mean()

print("\n=== 6D ===")
print(f"Somma TO_WITHOUTTAX TRS (clienti 2021, pre-2021): {sum_trs:,.0f} EUR")
print(f"Somma TO_FULL_HIST Aggregated 2021:               {sum_agg:,.0f} EUR")
print(f"Ratio TRS/AGG: {ratio:.3f} => {'COERENTE' if 0.5 <= ratio <= 2.0 else 'DISCREPANZA'}")
print(f"Spesa media/cliente TRS: {mean_trs:,.0f} EUR vs AGG: {mean_agg:,.0f} EUR")
print(f"N. medio TRS/cliente: {mean_n_trs:.1f}")

nb_cols = [c for c in agg_2021.columns if c.startswith('NB_TRS')]
if nb_cols:
    for col in nb_cols[:3]:
        print(f"  {col}: media={agg_2021[col].mean():.1f}, mediana={agg_2021[col].median():.1f}")

pd.DataFrame([{
    'Sum_TRS': round(sum_trs, 0), 'Sum_AGG': round(sum_agg, 0),
    'Ratio_TRS_AGG': round(ratio, 3),
    'Mean_spend_TRS': round(mean_trs, 0), 'Mean_spend_AGG': round(mean_agg, 0),
    'Mean_n_trs_per_client': round(mean_n_trs, 1),
}]).to_csv(OUT_TABLES / 'sanity_check_aggregates.csv', index=False)

print_finding("SANITY CHECK AGGREGATI",
    f"Somma TRS: {sum_trs:,.0f} EUR vs AGG: {sum_agg:,.0f} EUR. "
    f"Ratio: {ratio:.3f} - {'COERENTE' if 0.5<=ratio<=2.0 else 'DISCREPANZA'}. "
    "Salvato: sanity_check_aggregates.csv."
)

# ===========================================================================
# Tabelle riepilogo finali
# ===========================================================================
qt = pd.DataFrame([
    {'Check': 'TRS_DATE range',         'Value': f"{trs['TRS_DATE'].min()} -- {trs['TRS_DATE'].max()}"},
    {'Check': 'N transazioni totali',   'Value': len(trs)},
    {'Check': 'Post-ultimo-snapshot',   'Value': int(len(trs_leak_last))},
    {'Check': '% post-snapshot',        'Value': f"{len(trs_leak_last)/len(trs)*100:.1f}%"},
    {'Check': 'SN NULL',                'Value': n_null},
    {'Check': '% SN NULL',              'Value': f"{n_null/len(trs)*100:.1f}%"},
    {'Check': 'SN placeholder',         'Value': PLACEHOLDER},
    {'Check': 'N placeholder',          'Value': n_ph},
    {'Check': 'TO_WITHOUTTAX < 0',      'Value': int(neg_mask.sum())},
    {'Check': '% TO_WITHOUTTAX < 0',    'Value': f"{neg_mask.mean()*100:.3f}%"},
    {'Check': 'ARTICLE_WWPRICE = 0',    'Value': int(zero_mask.sum())},
    {'Check': 'Categorie mancanti Art', 'Value': str(sorted(cats_missing))},
    {'Check': 'Articoli cat. mancanti', 'Value': len(art_miss)},
])
qt.to_csv(OUT_TABLES / 'quality_transactions.csv', index=False)

qcc = pd.DataFrame([
    {'Dataset': 'CRC', 'Check': 'CREATION_DATE range', 'Value': f"{crc['CREATION_DATE'].min()} -- {crc['CREATION_DATE'].max()}"},
    {'Dataset': 'CRC', 'Check': 'CLIENT_ID unici',      'Value': len(crc_clients)},
    {'Dataset': 'CRC', 'Check': 'In snapshot 2021',     'Value': len(crc_in_2021)},
    {'Dataset': 'CRC', 'Check': '% clienti 2021',       'Value': f"{len(crc_in_2021)/len(clients_2021)*100:.1f}%"},
    {'Dataset': 'CRC', 'Check': 'Duration missing diagnosi', 'Value': diagnosi if dur_col else 'N/A'},
    {'Dataset': 'CCP', 'Check': 'CREATION_DATE range', 'Value': f"{ccp['CREATION_DATE'].min()} -- {ccp['CREATION_DATE'].max()}"},
    {'Dataset': 'CCP', 'Check': 'CLIENT_ID unici',      'Value': len(ccp_clients)},
    {'Dataset': 'CCP', 'Check': 'In snapshot 2021',     'Value': len(ccp_in_2021)},
    {'Dataset': 'CCP', 'Check': '% clienti 2021',       'Value': f"{len(ccp_in_2021)/len(clients_2021)*100:.1f}%"},
    {'Dataset': 'CCP', 'Check': 'Violazioni SALE>CREATION', 'Value': violations},
])
qcc.to_csv(OUT_TABLES / 'quality_crc_ccp.csv', index=False)

print("\n=== TABELLE SALVATE ===")
for f in sorted(OUT_TABLES.iterdir()):
    print(f"  {f.name}")
print("\n=== ANALISI FASE 5+6 COMPLETATA ===")
