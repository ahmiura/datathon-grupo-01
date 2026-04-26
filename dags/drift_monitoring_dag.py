import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'grupo_01_mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Esta DAG monitora anomalias no mercado de Segunda a Quinta-feira às 22:00. 
# (Sexta-feira fica reservada para o Continuous Training na DAG de MLOps).
with DAG(
    'drift_detection_and_retraining',
    default_args=default_args,
    description='DAG para detectar Data Drift no meio da semana para retreinos emergenciais',
    schedule_interval='0 22 * * 1-4',
    catchup=False,
    tags=['monitoring', 'mlops', 'evidently'],
) as dag:

    check_drift_task = BashOperator(
        task_id='run_drift_detection',
        bash_command='cd /app && python monitoring/drift.py',
    )