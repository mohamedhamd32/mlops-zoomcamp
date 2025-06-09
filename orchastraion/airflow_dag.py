from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 3, 1),
}

with DAG(
    dag_id="orch_hw3",
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    description="MLOps Homework 3 DAG",
) as dag:

    run_pipeline = BashOperator(
        task_id="run_pipeline_script",
        bash_command="python3 /root/my_dags/nyc_pipeline.py"
    )

    run_pipeline
