"""
DAG de Orquestração MLOps (Apache Airflow)
Objetivo: Automatizar o Retreinamento do Modelo LSTM e a Avaliação de Qualidade do Agente LLM.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Configurações padrão da DAG
default_args = {
    'owner': 'grupo-01',
    'depends_on_past': False,
    'email_on_failure': False, # Em produção, enviaríamos alerta para o Slack/Teams
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definição da DAG: Roda toda sexta-feira às 23:00 (após o fechamento da B3)
with DAG(
    'datathon_mlops_continuous_training',
    default_args=default_args,
    description='Pipeline de MLOps para Treinamento LSTM e Avaliação RAGAS',
    schedule_interval='0 23 * * 5',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'datathon', 'llm', 'lstm'],
) as dag:

    # Tarefa 1: Atualizar Feature Store de forma Incremental (Resolve GAP 03)
    update_feature_store_task = BashOperator(
        task_id='update_feature_store',
        bash_command='cd /app && python -c "from src.features.feature_store import update_feature_store; from src.config import TICKER, START_DATE, END_DATE; update_feature_store(TICKER, START_DATE, END_DATE)"',
    )

    # Tarefa 2: Retreinar o modelo de Deep Learning (LSTM) e registrar no MLflow
    # Em um ambiente real conteinerizado, o Airflow pode usar o DockerOperator ou KubernetesPodOperator
    train_model_task = BashOperator(
        task_id='train_lstm_model',
        bash_command='cd /app && python src/models/train.py',
    )

    # Tarefa 2.5: Forçar a recarga do modelo na API (Zero Downtime)
    reload_model_task = BashOperator(
        task_id='reload_api_model',
        bash_command='curl -X POST http://api:8000/reload-model',
    )

    # Tarefa 3: Avaliar a qualidade das respostas do Agente LLM usando RAGAS (LLM-as-a-judge)
    evaluate_llm_task = BashOperator(
        task_id='evaluate_agent_ragas',
        bash_command='cd /app && python evaluation/ragas_eval.py',
    )

    # Tarefa 4: Atualizar o Banco Vetorial (FAISS) com novos PDFs (Integração RAG)
    update_vector_db_task = BashOperator(
        task_id='update_rag_context',
        bash_command='cd /app && python -c "from src.agent.rag_pipeline import update_vector_store; update_vector_store()"',
    )

    # Tarefa 5: Avaliar métricas de negócio (Tom, Concisão, Risco) usando LLM-as-a-judge
    evaluate_business_task = BashOperator(
        task_id='evaluate_business_metrics',
        bash_command='cd /app && python evaluation/llm_judge.py',
    )

    # Definindo a ordem de execução do Pipeline (Dependências)
    update_feature_store_task >> train_model_task >> reload_model_task >> update_vector_db_task >> evaluate_llm_task >> evaluate_business_task