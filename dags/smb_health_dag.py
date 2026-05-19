from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "smb_health_pipeline",
    default_args=default_args,
    description="SMB health scoring pipeline: ingest → dbt → score → dashboard",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fintech", "smb", "health-score"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_synthetic_data",
        bash_command="python /opt/airflow/generate/synthetic_data.py",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /opt/airflow/smb_health_dbt && dbt run --profiles-dir .",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/airflow/smb_health_dbt && dbt test --profiles-dir .",
    )

    train_model = BashOperator(
        task_id="train_risk_model",
        bash_command="python /opt/airflow/models/risk_model/train.py",
    )

    generate_data >> dbt_run >> dbt_test >> train_model
