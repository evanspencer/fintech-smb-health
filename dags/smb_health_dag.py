import logging

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

with DAG(
    dag_id="folio_smb_health_pipeline",
    description="Folio SMB health pipeline: dbt → train → score → validate",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["folio", "smb", "health", "dbt", "xgboost"],
    default_args={
        "owner": "airflow",
        "retries": 1,
    },
) as dag:

    # ── task 1: dbt build ────────────────────────────────────────────────────
    run_dbt = BashOperator(
        task_id="run_dbt",
        bash_command=(
            "cd /opt/airflow/smb_health_dbt && "
            "dbt deps && "
            "dbt build --profiles-dir ."
        ),
    )

    # ── task 2: train xgboost model ──────────────────────────────────────────
    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && python3 models/risk_model/train.py",
    )

    # ── task 3: score all companies ──────────────────────────────────────────
    score_companies = BashOperator(
        task_id="score_companies",
        bash_command="cd /opt/airflow && python3 models/risk_model/score.py",
    )

    # ── task 4: validate pipeline outputs ───────────────────────────────────
    def _validate_outputs(**_):
        import duckdb

        con = duckdb.connect("/opt/airflow/data/smb_health.duckdb", read_only=True)

        n = con.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
        if n != 1000:
            con.close()
            raise AirflowException(
                f"model_scores has {n} rows — expected 1,000. "
                "score_companies may not have run correctly."
            )

        null_count = con.execute(
            "SELECT COUNT(*) FROM model_scores WHERE predicted_segment IS NULL"
        ).fetchone()[0]
        if null_count > 0:
            con.close()
            raise AirflowException(
                f"model_scores contains {null_count} NULL predicted_segment values."
            )

        at_risk_count = con.execute(
            "SELECT COUNT(*) FROM model_scores WHERE predicted_segment = 'at_risk'"
        ).fetchone()[0]
        if at_risk_count == 0:
            con.close()
            raise AirflowException(
                "No companies scored as at_risk — model output looks degenerate."
            )

        mart_n = con.execute("SELECT COUNT(*) FROM mart_company_health").fetchone()[0]
        if mart_n != 1000:
            con.close()
            raise AirflowException(
                f"mart_company_health has {mart_n} rows — expected 1,000. "
                "dbt build may not have completed successfully."
            )

        con.close()
        log.info(
            "✅ Folio pipeline validation passed — "
            "%d companies scored, %d flagged at-risk",
            n, at_risk_count,
        )

    validate_outputs = PythonOperator(
        task_id="validate_outputs",
        python_callable=_validate_outputs,
    )

    # ── dependencies ─────────────────────────────────────────────────────────
    run_dbt >> train_model >> score_companies >> validate_outputs
