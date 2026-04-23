import os
import logging
import pandas as pd
import mlflow
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.data.data_loader import fetch_financial_data
from src.models.train import train

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
    
    # Referência: Simula a janela de dados onde o modelo LSTM foi treinado
    ref_df = fetch_financial_data(ticker="PETR4.SA", start="2023-01-01", end="2023-12-31")
    ref_data = add_derived_features(ref_df)
    
    # Atual: Simula os dados recentes que o modelo está recebendo em Produção
    # Em um pipeline MLOps Nível 3, isso seria puxado do banco de dados da API (Feature Store / Logs)
    current_df = fetch_financial_data(ticker="PETR4.SA", start="2024-01-01", end="2024-06-01") 
    current_data = add_derived_features(current_df)

    if ref_data.empty or current_data.empty:
        logger.error("Não foi possível obter dados para a análise de drift. Abortando.")
        return

    # Selecionamos as variáveis numéricas cruciais para a previsão de séries temporais
    features = ['Close', 'Volume', 'Returns', 'Volatility']
    
    logger.info("Calculando Population Stability Index (PSI) e métricas de Drift via Evidently...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_data[features], current_data=current_data[features])

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
        mlflow.log_param("reference_start", "2023-01-01")
        mlflow.log_param("reference_end", "2023-12-31")
        mlflow.log_param("current_start", "2024-01-01")
        mlflow.log_param("current_end", "2024-06-01")
        mlflow.log_metric("dataset_drift_detected", int(dataset_drift))
        mlflow.log_artifact(report_path)
    
    if dataset_drift:
        logger.warning("🚨 ALERTA: Data Drift detectado na distribuição de preços/volatilidade!")
        logger.info("🔄 Iniciando retreinamento automático (Champion-Challenger) com a Feature Store atualizada...")
        train()
        logger.info("Notificando a API para recarregar o novo modelo em memória...")
        try:
            requests.post("http://api:8000/reload-model")
            logger.info("✅ API atualizada com sucesso (Zero Downtime).")
        except Exception as e:
            logger.error(f"Erro ao notificar a API: {e}")
    else:
        logger.info("✅ Distribuição de dados estável. Nenhum drift estrutural significativo detectado.")

if __name__ == "__main__":
    detect_drift()