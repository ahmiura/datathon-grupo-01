"""
DAG de Orquestração MLOps (Apache Airflow)
Objetivo: Automatizar o Retreinamento do Modelo LSTM e a Avaliação de Qualidade do Agente LLM.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Configurações padrão da DAG
default_args = {
    'owner': 'grupo-01',
    'depends_on_past': False,
    'email_on_failure': False, # Em produção, enviaríamos alerta para o Slack/Teams
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def evaluate_champion_logic():
    """Compara o modelo treinado (Challenger) com o modelo em produção (Champion)."""
    import os
    import json
    import shutil
    import logging
    
    logger = logging.getLogger(__name__)
    MODELS_DIR = '/app/models'
    challenger_metrics_path = os.path.join(MODELS_DIR, 'challenger_metrics.json')
    champion_metrics_path = os.path.join(MODELS_DIR, 'champion_metrics.json')
    
    if not os.path.exists(challenger_metrics_path):
        logger.warning("Nenhum modelo Challenger encontrado.")
        return 'update_rag_context'
        
    with open(challenger_metrics_path, 'r') as f:
        challenger_rmse = json.load(f).get('rmse', float('inf'))
        
    champion_rmse = float('inf')
    if os.path.exists(champion_metrics_path):
        with open(champion_metrics_path, 'r') as f:
            champion_rmse = json.load(f).get('rmse', float('inf'))
            
    logger.info(f"🏆 Champion RMSE: {champion_rmse:.4f} | ⚔️ Challenger RMSE: {challenger_rmse:.4f}")
    
    if challenger_rmse < champion_rmse:
        logger.info("🚀 Promovendo Challenger para produção...")
        shutil.copy(os.path.join(MODELS_DIR, "lstm_model_challenger.pth"), os.path.join(MODELS_DIR, "lstm_model.pth"))
        shutil.copy(os.path.join(MODELS_DIR, "challenger_hyperparameters.json"), os.path.join(MODELS_DIR, "model_hyperparameters.json"))
        if os.path.exists(os.path.join(MODELS_DIR, "scaler_challenger.joblib")):
            shutil.copy(os.path.join(MODELS_DIR, "scaler_challenger.joblib"), os.path.join(MODELS_DIR, "scaler.joblib"))
        shutil.copy(challenger_metrics_path, champion_metrics_path)
        return 'reload_api_model'
    
    logger.info("🛑 Mantendo Champion atual.")
    return 'update_rag_context'

# Definição da DAG: Roda toda sexta-feira às 23:00 (após o fechamento da B3 e sem colisão com a DAG de Drift)
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
        bash_command='cd /app && python -c "from src.features.feature_store import update_feature_store; from src.config import TICKER, START_DATE; from datetime import datetime; update_feature_store(TICKER, START_DATE, datetime.now().strftime(\'%Y-%m-%d\'))"',
    )

    # Tarefa 2: Retreinar o modelo de Deep Learning (LSTM) e registrar no MLflow
    # Em um ambiente real conteinerizado, o Airflow pode usar o DockerOperator ou KubernetesPodOperator
    train_model_task = BashOperator(
        task_id='train_lstm_model',
        bash_command='cd /app && python src/models/train.py',
    )

    # Tarefa 2.2: Avaliar se o novo modelo (Challenger) é melhor que o atual (Champion)
    evaluate_champion_task = BranchPythonOperator(
        task_id='evaluate_champion',
        python_callable=evaluate_champion_logic,
    )

    # Tarefa 2.5: Forçar a recarga do modelo na API (Zero Downtime)
    reload_model_task = BashOperator(
        task_id='reload_api_model',
        bash_command='python -c "import requests; requests.post(\'http://api:8000/reload-model\')"',
    )

    # Tarefa 3: Avaliar a qualidade das respostas do Agente LLM usando RAGAS (LLM-as-a-judge)
    evaluate_llm_task = BashOperator(
        task_id='evaluate_agent_ragas',
        bash_command='cd /app && python evaluation/ragas_eval.py',
    )

    # Tarefa 4: Atualizar o Banco Vetorial (FAISS) com novos PDFs (Integração RAG)
    update_vector_db_task = BashOperator(
        task_id='update_rag_context',
        bash_command='cd /app && python -c "from src.agent.rag_pipeline import update_vector_store; update_vector_store()" && python -c "import requests; requests.post(\'http://api:8000/reload-rag\')"',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # Tarefa 5: Avaliar métricas de negócio (Tom, Concisão, Risco) usando LLM-as-a-judge
    evaluate_business_task = BashOperator(
        task_id='evaluate_business_metrics',
        bash_command='cd /app && python evaluation/llm_judge.py',
    )

    # Fluxo de Dependências do Pipeline (Champion/Challenger)
    update_feature_store_task >> train_model_task >> evaluate_champion_task
    evaluate_champion_task >> reload_model_task >> update_vector_db_task
    evaluate_champion_task >> update_vector_db_task
    update_vector_db_task >> evaluate_llm_task >> evaluate_business_task