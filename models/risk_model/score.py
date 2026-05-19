"""
Scoring script — called by Airflow on each pipeline run.

Loads mart_company_health from DuckDB, scores all companies with the
trained XGBoost pipeline, writes results back to DuckDB as model_scores,
and prints a summary of accuracy and any misclassifications.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import duckdb
import joblib
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent.parent
DB_PATH    = ROOT / "data" / "smb_health.duckdb"
MODEL_PATH = Path(__file__).resolve().parent / "smb_health_model.pkl"

NUMERIC_FEATURES = [
    "txn_failure_rate",
    "missed_payment_rate",
    "late_payment_rate",
    "avg_days_late",
    "max_days_late",
    "avg_retry_count",
    "spend_trend",
    "credit_utilization",
    "total_spend",
    "avg_txn_amount",
    "unique_merchants",
    "unique_mcc_categories",
    "spend_last_30d",
    "spend_last_90d",
    "spend_last_180d",
]

CATEGORICAL_FEATURES = [
    "industry",
    "company_size",
    "region",
    "tenure_bucket",
    "dominant_payment_method",
    "most_common_failure_reason",
]

CLASS_NAMES = {0: "healthy", 1: "watch", 2: "at_risk"}


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    df  = con.execute("SELECT * FROM mart_company_health").df()

    pipeline = joblib.load(MODEL_PATH)

    X           = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predictions = pipeline.predict(X)
    probas      = pipeline.predict_proba(X)

    scores = pd.DataFrame({
        "company_id":          df["company_id"],
        "company_name":        df["company_name"],
        "health_segment":      df["health_segment"],
        "predicted_segment":   pd.Series(predictions).map(CLASS_NAMES),
        "correct":             df["health_segment"] == pd.Series(predictions).map(CLASS_NAMES),
        "score_proba_healthy": probas[:, 0].round(4),
        "score_proba_watch":   probas[:, 1].round(4),
        "score_proba_at_risk": probas[:, 2].round(4),
    })

    con.execute("CREATE OR REPLACE TABLE model_scores AS SELECT * FROM scores")
    con.close()

    accuracy     = scores["correct"].mean()
    misclassified = scores[~scores["correct"]]

    print(f"\n  Scored {len(scores)} companies  |  accuracy: {accuracy:.1%}")

    if len(misclassified) == 0:
        print("  All companies correctly classified.")
    else:
        print(f"\n  Misclassified ({len(misclassified)}):")
        print(f"  {'Company':<32}  {'Actual':<12}  {'Predicted':<12}  {'P(actual)':>10}")
        print("  " + "─" * 72)
        for _, row in misclassified.sort_values("health_segment").iterrows():
            actual    = row["health_segment"]
            proba_col = f"score_proba_{actual}"
            p_actual  = row[proba_col] if proba_col in row else float("nan")
            print(f"  {row['company_name']:<32}  {actual:<12}  {row['predicted_segment']:<12}  {p_actual:>10.4f}")


if __name__ == "__main__":
    main()
