# Folio — SMB Financial Health Scoring

An end-to-end analytics engineering project that models a B2B fintech platform scoring the financial health of small and medium businesses (SMBs) from their payment behavior. Identifies at-risk customers before they churn or default.

---

## The Problem

B2B fintech platforms serving hundreds or thousands of SMBs can't manually monitor every customer's payment trajectory. Risk signals — rising transaction failure rates, increasing days-late on invoices, declining spend — often appear weeks before a customer churns or defaults. Folio automates that monitoring: it ingests raw payment and transaction data, engineers behavioral features via dbt, and runs an XGBoost classifier to score every company in the portfolio daily.

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Synthetic data | Python + Faker | 1,000 SMBs · ~80k transactions · ~25k payments |
| Warehouse | DuckDB | Local analytical warehouse, no cloud cost |
| Transforms | dbt Core | Staging → intermediate → mart models with tests |
| Orchestration | Apache Airflow | DAG wiring data gen → dbt → scoring |
| ML scoring | XGBoost + scikit-learn | Multiclass health classifier + LogReg baseline |
| Dashboard | Streamlit + Plotly | Portfolio Overview · Company Explorer · Signal Analysis |

---

## Project Structure

```
fintech-smb-health/
│
├── generate/
│   └── synthetic_data.py          # Synthetic data generator (latent stress scores,
│                                  #   lifecycle phases, gamma-distributed amounts)
│
├── data/
│   ├── raw/                       # Generated CSVs (gitignored — run generator to reproduce)
│   │   ├── companies.csv
│   │   ├── transactions.csv
│   │   └── payments.csv
│   └── smb_health.duckdb          # DuckDB warehouse (gitignored)
│
├── smb_health_dbt/                # dbt project
│   ├── models/
│   │   ├── staging/               # stg_companies, stg_transactions, stg_payments
│   │   ├── intermediate/          # int_company_payment_behavior,
│   │   │                          #   int_company_transaction_behavior
│   │   └── marts/                 # mart_company_health (wide feature table),
│   │                              #   mart_daily_volume, mart_payment_failures
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── packages.yml               # dbt_utils
│
├── models/risk_model/
│   ├── train.py                   # XGBoost pipeline training + evaluation + serialization
│   ├── score.py                   # Batch scoring → model_scores table in DuckDB
│   └── model_metadata.json        # Held-out test-set metrics written by train.py
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard (3 pages)
│
├── dags/
│   └── smb_health_dag.py          # Airflow DAG
│
├── docker-compose.yml             # Airflow + Postgres
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python3 generate/synthetic_data.py
```

Writes `data/raw/companies.csv`, `transactions.csv`, `payments.csv` (~80k+ rows total).

### 3. Build the dbt pipeline

```bash
cd smb_health_dbt
dbt deps          # installs dbt_utils
dbt build         # seeds → staging → intermediate → marts + runs all 22 tests
cd ..
```

### 4. Train the model

```bash
python3 models/risk_model/train.py
```

Trains XGBoost and a LogReg baseline on `mart_company_health`, prints evaluation metrics, saves `smb_health_model.pkl` and `model_metadata.json`.

### 5. Score the portfolio

```bash
python3 models/risk_model/score.py
```

Writes scored predictions to the `model_scores` table in DuckDB.

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## Model Performance

Trained on a stratified 80/20 split of 1,000 companies.

| Metric | Held-out test set (n=200) |
|---|---|
| Accuracy | **89.5%** |
| Macro F1 | **0.849** |
| AUC-ROC | **0.979** |

The `watch` segment is hardest to classify — companies near the healthy/at-risk boundary exhibit naturally ambiguous signals, which is by design (see below).

---

## Data Design — Probabilistic Labels

The original version of this project achieved 100% model accuracy because health labels were derived deterministically from the same behavioral signals used as features. Real-world risk scoring doesn't work that way.

The current generator assigns labels probabilistically via a **latent stress score** (beta-distributed per company):

| Stress band | P(healthy) | P(watch) | P(at_risk) |
|---|---|---|---|
| Low  (< 0.35) | 85% | 12% | 3% |
| Mid  (0.35–0.65) | 20% | 60% | 20% |
| High (> 0.65) | 5% | 20% | 75% |

This creates genuine label noise at the boundaries — the same behavioral profile can be labeled `watch` or `healthy` depending on the draw — which forces the model to learn probabilistic estimates rather than memorize a lookup table. The result is a realistic 82–92% accuracy range on unseen data.

Companies also carry a **lifecycle phase** (stable / declining / recovering / volatile) that modulates spend trajectory and failure rates over time, creating temporal signal variation that mirrors real portfolio behavior.

---

## Dashboard Pages

**Portfolio Overview** — risk tier distribution, health score histogram, bottom-15 attention list with key signals.

**Company Explorer** — filterable table (industry, tier, size, region) with a drill-down detail panel per company: health score gauge, segment probability bar, and signals vs portfolio average.

**Signal Analysis** — feature importance chart, daily transaction volume by tier, failure reason breakdown, cohort box plots, and model performance metrics loaded live from `model_metadata.json`.

---

## Screenshots

_Coming soon._

---

## Airflow Orchestration

`dags/smb_health_dag.py` wires the full pipeline as a DAG:
`generate_data` → `dbt_build` → `train_model` → `score_portfolio`

Bring up the Airflow environment with:

```bash
docker-compose up -d
```
