import os
import logging
import pandas as pd
import mlflow
import requests
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import json
import shutil
from src.data.data_loader import fetch_financial_data
from src.models.train import train
from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas como Retornos e Volatilidade para análise de drift."""
    if 'Close' not in df.columns or df.empty:
        return pd.DataFrame()
    
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    return df.dropna()

def detect_drift():
    """
    Compara os dados de Treino (Referência) com os de Inferência (Atuais) 
    para detectar Data/Concept Drift na distribuição de preços.
    """
    logger.info("Coletando dados de Referência (Treino) e Atuais (Produção)...")
    
    # Define janelas de tempo dinâmicas baseadas na data de execução da DAG
    hoje = datetime.now()
    current_end = hoje.strftime('%Y-%m-%d')
    current_start = (hoje - timedelta(days=60)).strftime('%Y-%m-%d')  # Últimos 60 dias como Produção
    
    ref_end = (hoje - timedelta(days=61)).strftime('%Y-%m-%d')
    ref_start = (hoje - timedelta(days=365+60)).strftime('%Y-%m-%d')  # 1 ano antes da janela de produção como Treino

    # Referência: Simula a janela de dados onde o modelo LSTM foi treinado
    ref_df = fetch_financial_data(ticker="PETR4.SA", start=ref_start, end=ref_end)
    
    # Atual: Simula os dados recentes que o modelo está recebendo em Produção
    current_df = fetch_financial_data(ticker="PETR4.SA", start=current_start, end=current_end) 

    if ref_df.empty or current_df.empty or 'Close' not in ref_df.columns or 'Close' not in current_df.columns:
        logger.error("Não foi possível obter dados para a análise de drift. Abortando.")
        return

    logger.info("Calculando Population Stability Index (PSI) e métricas de Drift via Evidently...")
    drift_report = Report(metrics=[DataDriftPreset()])
    # Focamos o drift apenas na feature 'Close', que é a única usada pelo modelo LSTM univariado.
    drift_report.run(reference_data=ref_df[['Close']], current_data=current_df[['Close']])

    # Salva o relatório gerado em formato HTML para visualização em Dashboards/Artifacts
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "data_drift_report.html")
    drift_report.save_html(report_path)
    logger.info(f"📊 Relatório de Drift salvo com sucesso em: {report_path}")

    # Extrai o resultado para integração com alertas automáticos (ex: Slack / Email / Airflow Sensor)
    drift_summary = drift_report.as_dict()
    dataset_drift = drift_summary['metrics'][0]['result']['dataset_drift']
    
    logger.info("Enviando resultados de Drift para o MLflow...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Datathon_Data_Drift")
    
    with mlflow.start_run(run_name="Evidently_Drift_Check"):
        mlflow.log_param("reference_start", ref_start)
        mlflow.log_param("reference_end", ref_end)
        mlflow.log_param("current_start", current_start)
        mlflow.log_param("current_end", current_end)
        mlflow.log_metric("dataset_drift_detected", int(dataset_drift))
        mlflow.log_artifact(report_path)
    
    if dataset_drift:
        logger.warning("🚨 ALERTA: Data Drift detectado na distribuição de preços/volatilidade!")
        logger.info("🔄 Iniciando retreinamento automático (Champion-Challenger) com a Feature Store atualizada...")
        train()
        
        # --- Avaliação Local do Champion/Challenger antes de notificar a API ---
        challenger_metrics_path = os.path.join(MODELS_DIR, 'challenger_metrics.json')
        champion_metrics_path = os.path.join(MODELS_DIR, 'champion_metrics.json')
        
        if os.path.exists(challenger_metrics_path):
            with open(challenger_metrics_path, 'r') as f:
                challenger_rmse = json.load(f).get('rmse', float('inf'))
                
            champion_rmse = float('inf')
            if os.path.exists(champion_metrics_path):
                with open(champion_metrics_path, 'r') as f:
                    champion_rmse = json.load(f).get('rmse', float('inf'))
                    
            if challenger_rmse < champion_rmse:
                logger.info(f"🚀 Drift Model (RMSE: {challenger_rmse:.4f}) superou o atual. Promovendo...")
                shutil.copy(os.path.join(MODELS_DIR, "lstm_model_challenger.pth"), os.path.join(MODELS_DIR, "lstm_model.pth"))
                shutil.copy(os.path.join(MODELS_DIR, "challenger_hyperparameters.json"), os.path.join(MODELS_DIR, "model_hyperparameters.json"))
                if os.path.exists(os.path.join(MODELS_DIR, "scaler_challenger.joblib")):
                    shutil.copy(os.path.join(MODELS_DIR, "scaler_challenger.joblib"), os.path.join(MODELS_DIR, "scaler.joblib"))
                shutil.copy(challenger_metrics_path, champion_metrics_path)
                
                logger.info("Notificando a API para recarregar o novo modelo em memória...")
                try:
                    requests.post("http://api:8000/reload-model")
                    logger.info("✅ API atualizada com sucesso (Zero Downtime).")
                except Exception as e:
                    logger.error(f"Erro ao notificar a API: {e}")
            else:
                logger.warning(f"🛑 Modelo retreinado por Drift (RMSE: {challenger_rmse:.4f}) foi pior que o atual. Abortando deploy.")
        else:
            logger.error("Falha ao ler métricas do retreinamento de Drift.")
    else:
        logger.info("✅ Distribuição de dados estável. Nenhum drift estrutural significativo detectado.")

if __name__ == "__main__":
    detect_drift()