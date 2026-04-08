# Claude Code — EDA Briefing: Cartier QTEM Data Challenge

## Obiettivo
Eseguire un'Exploratory Data Analysis (EDA) strutturata in 8 fasi su 6 dataset Cartier.
L'obiettivo finale è costruire modelli di **short-term spend prediction** e **Customer Lifetime Value (CLV)**.
Tutti gli script esplorativi devono essere **read-only sui dati originali** — nessuna trasformazione in questa fase.

---

## Struttura dei Dataset

| Dataset             | Righe       | Colonne | Chiave primaria   | Note                              |
|---------------------|-------------|---------|-------------------|-----------------------------------|
| Aggregated_data     | 1.283.169   | 82      | CLIENT_ID?        | Dataset principale, molto sparse  |
| Transactions        | 916.957     | 14      | —                 | Storico transazioni               |
| Clients             | 412.571     | 9       | CLIENT_ID         | Anagrafica clienti                |
| CRC                 | 153.949     | 5       | APPOINTMENT_ID    | Appuntamenti con advisor          |
| CCP                 | 44.280      | 6       | —                 | Certificati di prodotto           |
| Articles            | 72.127      | 6       | ARTICLE_ID        | Catalogo prodotti                 |
| savings_rate        | 649         | 2       | Date              | Tasso risparmio mensile (esterno) |

---

## Problemi Noti da Investigare (priorità alta)

### 🚨 CRITICO — Aggregated_data: probabile parsing error del CSV
Dai summary generati, le colonne mostrano valori completamente sbagliati rispetto a quanto atteso:
- `DATE_TARGET` contiene valori tipo `country_of_residency_79` (dovrebbe essere una data)
- `TARGET_3Y` contiene valori tipo `Residency_Market_7` (dovrebbe essere un valore numerico di spend)
- `TARGET_5Y` contiene valori tipo `gender_2` (dovrebbe essere un valore numerico di spend)
- Molte colonne `ALL_*` mostrano valori con virgolette iniziali spurie e mix di tipi diversi

**Ipotesi**: il file CSV usa la virgola come delimitatore ma alcune colonne stringa (sequenze di prodotti acquistati, liste di date) contengono virgole non quotate, causando uno shift dei valori nelle colonne successive.

**Azione richiesta**: caricare il file con `pd.read_csv()` e stampare le prime righe raw per diagnosticare il problema. Verificare se serve `quotechar`, `escapechar`, o un parsing personalizzato.

### ⚠️ CLIENT_ID — type mismatch tra dataset
- In `Aggregated_data`: `CLIENT_ID` è `float64` (con 8 valori mancanti)
- In `Clients`: `CLIENT_ID` è `object` (stringa hash tipo `8d5631f1da54ffc16462`)
- In `Transactions`: `CLIENT_ID` è `object` (stringa hash)

Questi due formati sembrano incompatibili — il join tra Aggregated_data e gli altri dataset potrebbe essere impossibile senza una chiave di mapping. Da investigare con priorità.

### ⚠️ Aggregated_data: row count anomalo
- `Aggregated_data` ha 1.283.169 righe vs `Clients` con 412.571 clienti unici.
- Rapporto ~3:1 → l'Aggregated_data ha probabilmente **più osservazioni per cliente** (es. snapshot per paese/mercato o per periodo temporale).
- Capire la granularità della riga è fondamentale prima di qualsiasi join o feature engineering.

### ⚠️ Transactions — SERIAL_NUMBER mancante
- 97.221 righe mancanti (10,6% del totale — non 1,8% come da documentazione iniziale).
- Il valore top per SERIAL_NUMBER è un singolo hash (`4e2e3377...`) con 16.741 occorrenze → probabile placeholder.
- Da verificare: il placeholder è usato in modo uniforme o solo in certi canali/categorie?

### ⚠️ Transactions — valori negativi in TO_WITHOUTTAX_EUR_CONST
- Min: -83.6 EUR, 422 valori NULL.
- I valori negativi probabilmente codificano resi o rettifiche contabili (non documentato nel Data Dictionary).
- `TRS_CATEG` contiene solo `Sale` (725.473) e `Repair` (191.484) — nessuna categoria `Return` nonostante la documentazione la menzioni.
- I resi potrebbero quindi essere codificati come `Sale` con `TO_WITHOUTTAX_EUR_CONST` negativo.

### ⚠️ Transactions — ARTICLE_WWPRICE a zero
- Prezzo minimo: 0.0 EUR. Da verificare quante righe hanno prezzo zero e se corrispondono a gift, repair o errori.

### ⚠️ Articles vs Transactions — mismatch categorie prodotto
- `Articles.PRODUCT_CATEGORY` ha 8 valori unici: ProductCategory_2, 3, 4, 5, 6, 7, 8, 9.
- `Transactions.CATEG` ha solo 3 valori: ProductCategory_2, 4, 9.
- Le categorie 3, 5, 6, 7, 8 non appaiono nelle transazioni. Motivazione sconosciuta — da investigare (periodo di vendita? esclusione deliberata? problema di join?).

### ⚠️ Clients — BIRTH_DATE quasi vuota
- 258.244 valori mancanti su 412.571 (62,6%).
- Date placeholder sospette: `1804-01-01` (73 occorrenze), `1970-01-01` (54 occorrenze), `1950-04-01` (235 occorrenze in FIRST_TRANSACTION_DATE).
- `AGE` in Aggregated_data ha valore `0.5` come minimo e `1.0` come massimo e media — sembra normalizzato o una proporzione, non un'età in anni.

### ⚠️ TARGET_3Y — 76.570 valori mancanti in Aggregated_data
- `TARGET_5Y` e `TARGET_10Y` non hanno missing.
- Capire perché TARGET_3Y ha missing mentre gli altri no (finestra temporale? clienti acquisiti di recente?).

---

## Variabili Target — Note Critiche

- `TARGET_3Y`, `TARGET_5Y`, `TARGET_10Y` = spend totale atteso nei successivi 3/5/10 anni dalla DATE_TARGET.
- Distribuzione attesa: **power-law estrema** (8% dei clienti → 41% revenue; VIC 2% → 45% acquisti globali).
- Richiedono **log-transformation** prima della modellazione.
- `DATE_TARGET` è il **cutoff anti-leakage**: nessuna feature deve contenere informazioni posteriori a questa data.

---

## Fasi EDA da Eseguire

```
Fase 1 — Schema Validation          → tipi, colonne, shape reale vs atteso
Fase 2 — Missing Values             → % missing per colonna, pattern MCAR/MAR/MNAR
Fase 3 — Distribuzione Target       → TARGET_3Y/5Y/10Y in scala lineare e log
Fase 4 — DATE_TARGET e Leakage      → distribuzione cutoff, verifica colonne temporali
Fase 5 — Quality Analysis specifica → problemi noti per dataset (v. sopra)
Fase 6 — Referential Integrity      → join coverage tra dataset, clienti orfani
Fase 7 — Outlier Analysis           → spend estremo, QTY_PDT anomale, prezzi sospetti
Fase 8 — Decision Map Cleaning      → tabella problema → soluzione → motivazione
```

---

## Convenzioni di Lavoro

- **Separazione assoluta EDA / Cleaning**: questo script è solo esplorativo. Nessuna modifica ai dati originali.
- **Output**: salvare plot in `./output/plots/`, tabelle summary in `./output/tables/`.
- **Logging**: ogni finding rilevante va stampato con un header chiaro (es. `=== FINDING: SERIAL_NUMBER ===`).
- **Lingua commenti**: italiano.

---

## Domande Chiave a cui l'EDA deve Rispondere

1. Il CSV di Aggregated_data ha un problema di parsing? Come si leggono correttamente le 82 colonne?
2. Qual è la granularità della riga in Aggregated_data — un record per cliente o più?
3. CLIENT_ID in Aggregated_data è una chiave numerica interna distinta dall'hash degli altri dataset?
4. I valori negativi di TO_WITHOUTTAX_EUR_CONST si bilanciano con transazioni positive dello stesso cliente?
5. Il placeholder SERIAL_NUMBER (`4e2e3377...`) è uniforme per categoria/canale o concentrato?
6. Perché 5 categorie di Articles non appaiono in Transactions?
7. Qual è la distribuzione di DATE_TARGET — è un valore fisso o varia per cliente/mercato?
8. TARGET_3Y missing corrisponde a clienti acquisiti dopo una certa data (finestra insufficiente)?

## Finding Fase 1 — Aggiornamenti critici

- **FALSO ALLARME**: nessun problema di parsing su Aggregated_Data. 
  Si carica correttamente con settings default + parse_dates=["DATE_TARGET"].
- **Row count reale**: 1.517.798 righe (non 1.283.169 come documentato).
- **CLIENT_ID**: object/hash stringa in tutti i dataset. Join diretto 
  possibile, copertura 100% su 412.571 clienti.
- **STRUTTURA PANEL DATA**: granularità CLIENT_ID × DATE_TARGET. 
  Ogni cliente ha fino a 6 snapshot triennali (2006, 2009, 2012, 2015, 
  2018, 2021). Non è un dataset cross-sectional.
- **DATE_TARGET**: 6 valori fissi (non variabile per cliente). 
  È il cutoff temporale dello snapshot, non un cutoff globale.

## Finding Fase 2 — Missing Values

- Missing strutturali (STDDEV_TIMELAPSE, ALL_REPAIR_*): non imputare, 
  trasformare in flag binarie HAS_MULTIPLE_PURCHASES e HAS_REPAIR_HISTORY
- AGE: missing decresce 94.6% (2006) → 63.2% (2021). MAR correlato 
  al periodo. Usabile solo su snapshot 2021, escludere dagli snapshot storici.
- APPOINTMENT_DURATION (CRC): 86.1% missing, causa sconosciuta. MNAR sospetto.
- 0 clienti con >50% missing su snapshot 2021 — nessuna esclusione necessaria.

## Finding Fase 3 — Target e Modellazione

- TARGET_3Y: 95.2% valori zero su snapshot 2021. Power-law estrema.
- Gini = 0.63. Top 1% clienti → 67.5% revenue totale.
- TARGET_5Y ≈ TARGET_10Y al 97.5% — usare solo TARGET_3Y e TARGET_5Y.
- DECISIONE MODELLO: Two-Part Model (Hurdle Model)
  - Parte 1: classificatore P(TARGET > 0)
  - Parte 2: regressore log(TARGET | TARGET > 0)
- Log-transform necessaria: skewness 34 → 0.03 dopo log1p sui non-zero.
- APERTO: definire metrica di valutazione adeguata per distribuzione 
  zero-inflated (non usare RMSE classico).

## Finding Fase 4 — Anti-Leakage

- Nessun leakage confermato. Tutte le colonne verificate sono SAFE.
- RECENCY, SENIORITY, ALL_PURCHASED_DATES, ALL_REPAIR_DATES: tutti 
  riferiti a DATE_TARGET, non alla data di estrazione.
- Il dataset è costruito correttamente rispetto al cutoff temporale.

## Errori documentazione corretti

- TARGET_3Y: 0 valori mancanti (non 76.570 come da nota precedente)
- Parsing Aggregated_Data: nessun problema (nota precedente era errata)

## Decisioni Modellazione — Two-Part Model (Hurdle Model)

### Struttura del modello
- **Parte 1 — Classificatore**: P(TARGET > 0) — identifica se il cliente spenderà
- **Parte 2 — Regressore**: E[log(TARGET) | TARGET > 0] — stima quanto spenderà
- **Predizione finale**: P(spende) × E[quanto spende | spende]

### Metriche di valutazione
- **Parte 1**: Precision-Recall AUC (metrica principale) + Recall sul top decile
  - NON usare accuracy: con 95% zeri un modello dummy ottiene 95% accuracy
  - NON usare AUC-ROC come metrica principale: instabile con imbalance estremo
- **Parte 2**: RMSE in log-space per il tuning + MAE in EUR dopo back-transform
  per la reportistica a Cartier
  - NON usare R² come metrica principale: instabile su distribuzioni skewed

### Strategia di validazione
- **Opzione A — Holdout temporale** (implementazione principale):
  train su snapshot 2006–2018, test su snapshot 2021 (412.571 clienti)
- **Opzione B — Walk-forward** (backtest opzionale): solo se tempo disponibile
  dopo consolidamento Opzione A
- **Regola critica**: snapshot 2021 completamente isolato fino alla valutazione
  finale — nessun tuning iperparametri basato sul validation set

### Target confermati
- Usare TARGET_3Y e TARGET_5Y
- Escludere TARGET_10Y: TARGET_5Y ≈ TARGET_10Y al 97.5% — non aggiunge informazione

## Decisioni Feature Engineering — Missing Values

### Trattamento per colonna
| Colonna | Trattamento |
|---|---|
| STDDEV/AVG/MIN_TIMELAPSE | Flag binaria HAS_MULTIPLE_PURCHASES (0/1). Valori originali solo per NB_TRS > 1 |
| ALL_REPAIR_* | Flag binaria HAS_REPAIR_HISTORY (0/1). Valori originali solo dove presenti |
| AGE | Solo snapshot 2021. Imputare mediana per mercato/genere + flag AGE_KNOWN (0/1) |
| APPOINTMENT_DURATION | Sospeso — verificare distribuzione missing per ORIGIN in Fase 5 (CRC). Nel frattempo: flag HAS_APPOINTMENT_DURATION (0/1) |

### Regola generale
Colonne con missing strutturale non si imputano — si trasformano in flag binarie.
Imputazione solo per missing MAR su colonne con <50% missing sullo snapshot rilevante.

### APPOINTMENT_DURATION — RISOLTO (Fase 5F)
- **MAR** — missing fortemente correlato a ORIGIN (std = 30.1% tra gruppi).
- ORIGIN=CRC: 0.03% missing. ORIGIN=Clienteling: 1.5%. ORIGIN=Web: 37.4%.
- ORIGIN=Phone/Email/Boutique/Hotline: 100% missing.
- **Decisione**: usare APPOINTMENT_DURATION solo da ORIGIN in {CRC, Clienteling, Web}.
  Per tutti gli altri: flag HAS_APPOINTMENT_DURATION=0.

---

## Finding Fase 5+6 — Quality Analysis e Referential Integrity

### 5A — Transactions: copertura temporale
- TRS_DATE va da 2000-01-03 a 2023-12-31 (non fino al 2025 come stimato).
- 63.683 transazioni (6.9%) con TRS_DATE > 2021-01-01 — escluse per snapshot 2021.
- **Regola operativa**: filtrare TRS_DATE <= DATE_TARGET prima di calcolare qualsiasi feature.

### 5B — SERIAL_NUMBER
- NULL: 97.221 (10.6%). Placeholder `4e2e3377...`: 16.741 (1.8%).
- Placeholder concentrato al 100% su Boutique. NULL su tutti i canali.
- NULL = principalmente vendite senza serial tracciato. Pattern distinto dal placeholder.

### 5C — Valori negativi TO_WITHOUTTAX_EUR_CONST
- **Solo 4 valori negativi** (0.0004%) — tutti Repair, non Sale.
- 422 NULL e 18.130 valori = 0 sono il vero problema.
- **Conclusione**: i negativi sono rettifiche su riparazioni. Impatto trascurabile.

### 5D — ARTICLE_WWPRICE a zero
- 41.716 righe (4.55%) con WWPRICE=0. Mix Repair (51.7%) e Sale (48.3%).
- TO_WITHOUTTAX > 0 in 23.544 casi anche quando WWPRICE=0.

### 5E — Categorie Articles non in Transactions
- Categorie mancanti: ProductCategory_3, 5, 6, 7, 8. Articoli: 21.499.
- **Zero overlap ARTICLE_ID** — non esistono in nessuna transazione.
- **Conclusione**: le macro-categorie Transactions (2, 4, 9) aggregano piu PRODUCT_CATEGORY.
  Le 5 categorie mancanti sono linee non vendute nel periodo coperto.

### 5F — CRC: APPOINTMENT_DURATION
- Coverage snapshot 2021: 28.036 clienti (6.8%). Copertura limitata.
- **Diagnosi: MAR** — std % missing per ORIGIN = 30.1%.
  ORIGIN CRC: 0.03% missing. Clienteling: 1.5%. Web: 37.4%. Phone/Email: 100%.

### 5G — CCP: copertura e consistenza date
- Coverage snapshot 2021: 5.786 clienti (1.4%) — insufficiente come feature standalone.
- 228 violazioni SALE_DATE > CREATION_DATE. SALE_DATE missing ~60% in entrambi i gruppi gift.

### 6A — Matrice Referential Integrity
- **Aggregated_2021 <-> Transactions <-> Clients**: 100% coverage bidirezionale. Join perfetto.
- **CRC**: copre 6.8% di Agg_2021. **CCP**: copre 1.4% di Agg_2021.
- Usare come segnali binari (HAS_CRC, HAS_CCP), non feature continue.

### 6B — Clienti orfani
- **0 clienti orfani** tra Transactions, Aggregated e Clients.
- CRC: 45.395 non in Agg_2021. CCP: 29.364 non in Agg_2021 (clienti post-2021).

### 6C — ARTICLE_ID coverage
- 81.4% di ARTICLE_ID in Transactions matchati in Articles. 5.378 orfani (18.6%).
- Orfani distribuiti su tutti gli anni e categorie — probabilmente articoli dismessi.

### 6D — Sanity check aggregati
- **PERFETTA COERENZA**: ratio TO_WITHOUTTAX TRS / TO_FULL_HIST Agg = 1.006.
- Spesa media/cliente: 4.472 EUR (TRS) vs 4.445 EUR (Agg). NB_TRS medio = 1.3.

### Implicazioni operative aggiuntive
- ARTICLE_ID orfani (18.6%): probabilmente articoli dismessi. 
  Feature Articles (WORLD_PRICE, FLAG_HE, FLAG_BRIDAL, FLAG_DIAMOND) 
  recuperabili via join su ARTICLE_ID — usare left join, non inner join.
- 228 violazioni SALE_DATE > CREATION_DATE in CCP: droppare nel cleaning.
- TO_WITHOUTTAX_EUR_CONST negativi (4 righe): droppare nel cleaning, 
  impatto trascurabile.

### CASO WWPRICE=0 con TO>0 — RISOLTO (Fase 7D)
- **3.386 Repair**: LEGITTIMO. Costo servizio riparazione senza articolo prezzato nel catalogo.
- **20.158 Sale**: INCERTO. Articoli legacy/custom non catalogati.
  Decisione cleaning: mantenere, aggiungere flag WWPRICE_MISSING=1.
  Non droppare: il TO_WITHOUTTAX contiene comunque il valore di vendita.

---

## Finding Fase 7 — Outlier Analysis

### 7A — Target outliers
- TARGET_3Y non-zero (19.919 clienti su 412.571 = 4.8%): p99=64k, p99.9=165k, max=503k EUR.
- Clienti oltre p99.9 (> 165k EUR): 20 — VIC reali, non rimuovere.
- Profilo top 50: SENIORITY media=80, RECENCY media=34, NB_TRS media=3.7, TO_FULL_HIST media=63k.
  Coerente con VIC: alta seniority, bassa recency (acquistano frequentemente).
- **ZERO violazioni TARGET_3Y > TARGET_5Y** — dataset logicamente coerente.

### 7B — Spend storico outliers
- TO_FULL_HIST p99.9: 159k EUR, max=624k EUR. Plausibile per VIC con decenni di acquisti.
- Overlap top TO_FULL_HIST (p99.9) e top TARGET_3Y (p99.9): solo 4 clienti (1%).
  ATTENZIONE: i clienti con spend storico massimo hanno spesso TARGET_3Y=0 (alta recency/anziani).
  Il modello NON deve usare solo TO_FULL_HIST per stimare il target futuro.
- TO_FULL_HIST=0 con TARGET_3Y>0: 0 casi. Dataset coerente.

### 7C — Frequency outliers
- NB_TRS_FULL_HIST: p99=6, p99.9=13, max=79. Distribuzione molto concentrata su 1-3 transazioni.
- Clienti con NB_TRS > 13: 410 (0.1%). TO_FULL_HIST medio 98k. Alta frequenza = alta spesa.
- QTY_PDT in Transactions: p99=1, max=70. Solo 56 transazioni con QTY_PDT > 10.
  Tutte Boutique, 91% ProductCategory_2 (accessori). Plausibile (acquisto multiplo stesso articolo).
- NB_TRS alto con TO_FULL_HIST basso: 0 casi. Nessuna anomalia "riparazioni ripetute a basso valore".

### 7D — Price outliers
- ARTICLE_WWPRICE max: 590.000 EUR (Article_113018, ProductCategory_4/Boutique). LEGITTIMO.
- WORLD_PRICE max: 5.500.000 EUR. Articoli > 100k EUR: 3.030 (4.2%), 94% FLAG_HE=1. LEGITTIMO.
- Articoli WORLD_PRICE < 1 EUR: 20, tutti ProductCategory_7 (non in Transactions). INCERTO.
- **RISOLTO**: WWPRICE=0 con TO>0 = 23.544 righe. Repair=LEGITTIMO, Sale=INCERTO/mantenere.

### 7E — Outliers demografici
- BIRTH_DATE pre-1900: 1.891. Top placeholder: 1804-01-01 (73), 1804-03-16 (12).
- Eta < 18 anni al 2021: 421. Eta > 100 anni: 1.993. Totale anomalie: 2.312 (0.56% dei clienti).
- FIRST_PURCHASE vs FIRST_TRANSACTION > 365gg: 7.243 (1.8%). Max 25.538 giorni (anomalo).
- SENIORITY: min=1, max=852. Nessuno = 0. Distribuzione pulita.

### 7F — CRC/CCP outliers
- APPOINTMENT_DURATION max: 4.350 minuti (72 ore). Solo 4 valori > 8h: ERRORE.
- Top cliente CRC: 3.207 interazioni (probabilmente account istituzionale o test).
- CCP violazioni SALE_DATE > CREATION_DATE: 228. Di cui 121 (<= 7gg, arrotondamento),
  94 (> 30gg, errori seri). Droppare le 94 serie nel cleaning.

### 7G — Tabella classificazione outliers
| Dataset | Pattern | N | Classificazione |
|---|---|---|---|
| Aggregated_2021 | TARGET_3Y > p99.9 | 20 | LEGITTIMO |
| Aggregated_2021 | TARGET_3Y > TARGET_5Y | 0 | OK |
| Aggregated_2021 | TO_FULL_HIST > p99.9 | 413 | LEGITTIMO |
| Transactions | QTY_PDT > 10 | 56 | INCERTO |
| Transactions | WWPRICE=0 TO>0 Repair | 3.386 | LEGITTIMO |
| Transactions | WWPRICE=0 TO>0 Sale | 20.158 | INCERTO - flag WWPRICE_MISSING |
| Articles | WORLD_PRICE > 100k | 3.030 | LEGITTIMO |
| Articles | WORLD_PRICE < 1 EUR | 20 | INCERTO |
| Clients | BIRTH_DATE anomale | 2.312 | ERRORE - droppare/nullify |
| CRC | APPOINTMENT_DURATION > 8h | 4 | ERRORE |
| CCP | SALE_DATE > CREATION_DATE | 228 | ERRORE (94 seri) |

### Implicazione critica per la modellazione (da 7B)
- TO_FULL_HIST NON è un predittore affidabile di TARGET_3Y.
  Overlap top spender storici / top target futuri: solo 1%.
- Feature predittive critiche: RECENCY, TO_PAST_3Y, ciclo di acquisto
  recente — non lo spend storico aggregato.
- Gli snapshot storici (2006-2015) sono preziosi per catturare
  l'evoluzione comportamentale nel tempo, non solo lo stato attuale.
- Nel Two-Part Model, la Parte 1 (classificatore) deve pesare
  fortemente le feature di attività recente rispetto allo storico totale.

---

## Fase 8 — Decision Map e Cleaning Pipeline (COMPLETATA)

### Dataset processati in data/processed/
| File | Righe | Colonne | Righe rimosse | Colonne aggiunte |
|---|---|---|---|---|
| Aggregated_Data_clean.csv | 1.517.798 | 85 | 0 | 3 |
| Transactions_clean.csv | 916.953 | 17 | 4 | 3 |
| Clients_clean.csv | 412.571 | 12 | 0 | 3 |
| CRC_clean.csv | 153.945 | 6 | 4 | 1 |
| CCP_clean.csv | 44.186 | 8 | 94 | 2 |
| supplementary_features.csv | 483.214 | 10 | — | 10 |

### Flag binarie create (10 totali)
- **Aggregated_Data**: AGE_KNOWN, HAS_MULTIPLE_PURCHASES, HAS_REPAIR_HISTORY
- **Transactions**: TO_WITHOUTTAX_IMPUTED, SERIAL_NUMBER_KNOWN, WWPRICE_MISSING
- **Clients**: BIRTH_DATE_VALID, PURCHASE_DATE_ANOMALY
- **CRC**: HAS_APPOINTMENT_DURATION
- **CCP**: SALE_DATE_VALID

### Supplementary features per CLIENT_ID (483.214 clienti)
- Da CRC: HAS_CRC_INTERACTION, N_CRC_INTERACTIONS, HAS_CLIENTELING, AVG_APPOINTMENT_DURATION
- Da CCP: HAS_CCP, N_CCP_PRODUCTS, HAS_GIFT_REGISTERED
- Da Clients: BIRTH_DATE_VALID, PURCHASE_DATE_ANOMALY

### Validazione post-cleaning: 23/23 check PASS
- Zero valori negativi in TO_WITHOUTTAX_EUR_CONST
- Zero placeholder SERIAL_NUMBER residui
- Zero BIRTH_DATE pre-1900 residue
- Zero APPOINTMENT_DURATION > 8h in CRC
- Zero CLIENT_ID null in Aggregated_Data
- Tutte le 10 flag binarie contengono solo valori 0/1
- TARGET_3Y/5Y/10Y immutati rispetto al raw

### Decisioni chiave dalla Decision Map (25 problemi documentati)
- **DROP**: CLIENT_ID null (8), TO_WITHOUTTAX<0 (4), CRC DURATION>8h (4), CCP SALE>CREATION>30gg (94)
- **FLAG senza drop**: AGE, STDDEV_TIMELAPSE, REPAIR_HISTORY, TO_WITHOUTTAX_IMPUTED,
  SERIAL_NUMBER, WWPRICE_MISSING, BIRTH_DATE, PURCHASE_DATE_ANOMALY, HAS_APPOINTMENT_DURATION
- **MANTENERE senza azione**: QTY_PDT>10, WORLD_PRICE<1EUR, CCP SALE>CREATION<=7gg
- **NON applicare in cleaning**: filtro TRS_DATE<=DATE_TARGET (da applicare in feature engineering)
- **JOIN regola**: LEFT JOIN tra Transactions e Articles (non INNER) per preservare orfani

### Script production-ready
- `scripts/cleaning.py`: funzioni indipendenti per dataset, `run_all_cleaning()` per pipeline completa
- `scripts/build_decision_map.py`: genera decision_map.csv
- `output/tables/cleaning_validation_report.csv`: 23 check con status PASS/FAIL


## Finding FE Step 1 — Stato Repository

### Colonne aggiuntive non documentate
- Clients_clean: MonoMultiMarket (mono/multi-mercato) — utile come feature
- Transactions_clean: nomi colonne reali sono CATEG, SUBCATEG, 
  Collection, PRODUCT_FUNCTION (non PDT_CATEG ecc. come da data dictionary)

### Feature da escludere per zero/near-zero variance
- MAX_PRICE_IN_BTQ: 100% zero — eliminare
- NB_TRS_BTQ: 100% zero — eliminare  
- TO_OTHER_HE: 99.99% zero — eliminare
- TO_CRC, TO_WEB, TO_MORE_10K: >99% zero — escludere dal modello

### ALL_PURCHASED_* e ALL_REPAIR_*
- 12 colonne, tutte stringhe raw comma-separated non parsate
- Richiedono str.split(',') + explode — escluse dal FE per ora
- Da considerare come lavoro futuro se tempo disponibile

### Supplementary features — rischio leakage
- supplementary_features.csv aggregato su tutto lo storico senza filtro
- Per sicurezza: usare solo HAS_CRC e HAS_CCP come flag binarie
- Escludere N_CRC_INTERACTIONS e AVG_APPOINTMENT_DURATION dal training

### data/raw/ 
- Cartella vuota nella repo — raw non copiati
- Cleaning script non rieseguibile senza raw
- Dataset processed esistono e sono corretti (19/03/2026)
## Finding FE Step 2 — Feature Engineering Completato (08/04/2026)

### Feature set finale
- **Train**: 1.105.227 righe × 92 colonne — snapshot 2006, 2009, 2012, 2015, 2018
- **Test**: 412.571 righe × 92 colonne — snapshot 2021 (isolato)
- **Feature totali**: 83 (esclusi CLIENT_ID, DATE_TARGET, target)
  - Aggregated: 56 colonne pre-calcolate
  - RFM_Transactions: 17 feature calcolate per snapshot
  - Articles: 6 feature (join LEFT su ARTICLE_ID)
  - Supplementary: 4 flag binarie (HAS_CRC, HAS_CCP, HAS_GIFT_REGISTERED, HAS_CLIENTELING)

### Validazione: 9/9 check PASS
- Nessuna colonna zero-variance
- BINARY_TARGET_3Y correttamente 0/1
- LOG_TARGET_3Y non negativo
- Missing media: 0.1% (solo colonne Articles: 1.8% per clienti senza Sales nel periodo)

### Top 5 feature predittive (correlazione |r| con TARGET_3Y, esclusi altri target)
1. TO_PAST_3Y (Aggregated): r=0.182
2. SPEND_PAST_3Y (RFM_Transactions): r=0.181  ← conferma che le feature RFM riproducono Aggregated
3. TOTAL_SPEND / TO_FULL_HIST: r=0.165
4. TO_BTQ: r=0.165
5. TO_JWL: r=0.151

### Implicazione critica: ridondanza RFM vs Aggregated
- Le feature calcolate da Transactions (SPEND_PAST_3Y) sono quasi identiche alle colonne
  di Aggregated_Data (TO_PAST_3Y). Correlazione attesa ~0.99.
- Nel modello finale: considerare di usare SOLO le colonne Aggregated ed escludere le
  feature RFM ridondanti, per ridurre il rischio di multicollinearità.
- Le feature RFM uniche (BOUTIQUE_RATIO, HOLIDAY_PURCHASE_RATIO, AVG_DAYS_BETWEEN_TRS)
  non hanno equivalenti in Aggregated — da mantenere.

### Decisioni implementazione
- TO_STDDEV_SPREAD aggiunto a ALWAYS_EXCLUDE (84.6% missing strutturale, stessa policy STDDEV_PRICE)
- AVG_DAYS_BETWEEN_TRS: NaN fill=0 per clienti con 1 sola transazione (nessun gap misurabile)
- Articles.csv: recuperato da percorso alternativo (data/raw/ vuota nel repo)
- Supplementary: solo 4 flag binarie sicure — esclusi N_CRC_INTERACTIONS e AVG_APPOINTMENT_DURATION

### Script e output
- `scripts/feature_engineering.py`: pipeline modulare, eseguibile end-to-end
- `notebooks/06_feature_engineering.ipynb`: notebook documentativo
- `data/features/`: 7 file CSV (train_features.csv e test_features.csv pronti per modeling)
- `output/tables/feature_correlations.csv`: correlazioni complete con TARGET_3Y
- `output/tables/feature_engineering_report.csv`: report statistiche feature set

## Finding Feature Selection (08/04/2026)

### Feature set finale (train_features_final.csv / test_features_final.csv)
- **Train**: 1.105.227 × 72 — snapshot 2006-2018
- **Test**: 412.571 × 72 — snapshot 2021 (isolato)
- **63 feature** (esclusi CLIENT_ID, DATE_TARGET, target)
- **20 feature rimosse** da train_features.csv (92 colonne → 72)
  - 13 near-zero variance (>99% zero)
  - 7 duplicati RFM con equivalente in Aggregated

### Coppie Aggregated-Aggregated r>0.95 (NON rimosse — decisione manuale richiesta)
| Coppia | r |
|---|---|
| TO_FULL_HIST <-> TO_BTQ | 0.9999 |
| TO_PAST_3Y_6Y <-> SPEND_3Y_6Y | 0.9961 |
| TO_10K_20K <-> QTY_PDT_10K_20K | 0.9852 |
| TO_5K_10K <-> QTY_PDT_5K_10K | 0.9829 |
| TO_20K_50K <-> QTY_PDT_20K_50K | 0.9746 |
| RECENCY <-> RECENCY_DAYS | 0.9549 |
| MAX_PRICE_PER_PDT <-> MAX_PRICE_PER_TRS | 0.9514 |
| MAX_PRICE_PER_PDT <-> MAX_SINGLE_SPEND | 1.0000 |

Nota: queste coppie sono mantenute — per un modello tree-based (XGBoost/LightGBM)
la multicollinearità non è un problema critico. Per regressione logistica/lineare
conviene rimuoverne uno per coppia prima del training.

### Colonne RFM aggiuntive rimosse rispetto a previsione
- FLAG_HE_RATIO_TRS: >99% zero nel train (era in RFM_KEEP ma inutilizzabile)
- RECENCY_DAYS: r=0.955 con RECENCY (Aggregated) — ridondante
- MAX_SINGLE_SPEND: r=1.0 con MAX_PRICE_PER_PDT (Aggregated) — identica

### Script
- `scripts/feature_selection.py`: pipeline modulare, eseguibile end-to-end
## Feature Engineering — Completato

### Dataset finali
- train_features.csv: 1.105.227 × 92 (snapshot 2006-2018)
- test_features.csv: 412.571 × 92 (snapshot 2021, isolato)
- 83 feature totali (esclusi id e target)

### Top 5 predittori per TARGET_3Y
1. TO_PAST_3Y / SPEND_PAST_3Y — r≈0.18
2. TOTAL_SPEND / TO_FULL_HIST — r≈0.165
3. TO_BTQ / TO_JWL — r≈0.15
4. NB_TRS_FULL_HIST — r≈0.149
5. MAX_ARTICLE_WORLD_PRICE — r≈0.116

### Decisione multicollinearità
- SPEND_PAST_3Y ≈ TO_PAST_3Y — tenere solo colonne Aggregated
- Feature RFM uniche da mantenere: BOUTIQUE_RATIO,
  HOLIDAY_PURCHASE_RATIO, AVG_DAYS_BETWEEN_TRS

### Feature escluse
- MAX_PRICE_IN_BTQ, NB_TRS_BTQ: zero variance
- TO_OTHER_HE: near-zero variance (99.99% zero)
- ALL_PURCHASED_*: stringhe raw non parsate — lavoro futuro
- Supplementary continue (N_CRC_INTERACTIONS, AVG_APPOINTMENT_DURATION):
  rischio leakage temporale — usate solo HAS_CRC e HAS_CCP

### Prossimo step: M2 — Short-Term Spend Prediction (modeling)

## Feature Engineering — Set Finale

### Dataset finali pronti per modeling
- train_features_final.csv: 1.105.227 × 72 (63 feature + 9 id/target)
- test_features_final.csv: 412.571 × 72 (isolato)
- 20 feature rimosse: 13 near-zero variance + 7 duplicati RFM

### Coppie ad alta correlazione — decisione per modeling
- TO_FULL_HIST ↔ TO_BTQ (r=0.9999): innocua per tree-based,
  rimuovere TO_BTQ per regressione logistica
- MAX_PRICE_PER_PDT ↔ MAX_PRICE_PER_TRS ↔ MAX_SINGLE_SPEND (r≈1.0):
  rimuovere MAX_PRICE_PER_PDT e MAX_PRICE_PER_TRS prima del modeling
  indipendentemente dall'algoritmo
- TO_*K ↔ QTY_PDT_*K (r=0.97-0.99): innocua per tree-based,
  rimuovere QTY_PDT_*K per regressione lineare/logistica

### Planned improvements — Feature Engineering
1. Parsing ALL_PURCHASED_* (sequenze prodotti) — alto impatto sul classificatore
2. Feature di interazione: SPEND_PAST_3Y/SENIORITY, SPEND_TREND×RECENCY_DAYS
3. Feature CRC con filtro temporale corretto per snapshot storici
4. Ciclo di acquisto individuale da ALL_PURCHASED_DATES