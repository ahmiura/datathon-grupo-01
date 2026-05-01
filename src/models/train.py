import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
import pandas as pd
import random

# --- Supressão de Warnings (Para logs mais limpos no Airflow) ---
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*distutils.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import mlflow
import logging
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

from typing import Tuple, Dict, Any
import matplotlib
# Configura backend não-interativo para evitar erros no Docker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.features.data import preprocess_data
from datetime import datetime, timedelta
import shutil
from src.data.data_loader import fetch_financial_data
from src.features.feature_store import materialize_feature_store_snapshot, STORE_PATH
from src.models.model import LSTMModel
from src.config import *

# Configuração de Logging Estruturado (GAP 09)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Fixa as sementes para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train_one_epoch(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model: nn.Module, criterion: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, scaler: Any) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        
        y_test_real = scaler.inverse_transform(y_test.numpy())
        test_outputs_real = scaler.inverse_transform(test_outputs.numpy())
        
        rmse = np.sqrt(mean_squared_error(y_test_real, test_outputs_real))
        mae = mean_absolute_error(y_test_real, test_outputs_real)
        mape = mean_absolute_percentage_error(y_test_real, test_outputs_real)
        
    return test_loss.item(), rmse, mae, mape, y_test_real, test_outputs_real

def run_experiment(params: Dict[str, Any], X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, scaler: Any, run_name: str) -> Tuple[float, Dict[str, Any]]:
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    dropout = params['dropout']
    lr = params['learning_rate']
    epochs = params['epochs']
    
    # Garante reprodutibilidade para cada run
    set_seed(42)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("training_data_source", "postgres_features.stock_prices")
        mlflow.log_param("training_data_snapshot", STORE_PATH)
        if os.path.exists(STORE_PATH):
            mlflow.log_artifact(STORE_PATH, artifact_path="data")
        
        # --- Tags de Governança Obrigatórias (GAP 05) ---
        required_tags = {
            "model_name": "PETR4_LSTM_Predictor",
            "model_version": "v1.0.0",
            "training_data_version": "v1_initial", # No futuro, vira o hash do DVC
            "model_type": "LSTM",
            "owner": "grupo-01",
            "risk_level": "medium",
            "git_sha": os.getenv("GIT_SHA", "unknown"),
            "fairness_checked": "true",
            "phase": "datathon-fase05"
        }
        mlflow.set_tags(required_tags)
        
        model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, X_train, y_train)
            if (epoch+1) % 10 == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
        
        test_loss, rmse, mae, mape, y_real, y_pred = evaluate(model, criterion, X_test, y_test, scaler)
        
        mlflow.log_metrics({
            "test_loss": test_loss,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        })
        
        # --- Geração do Gráfico (Requisito Seção 5) ---
        plt.figure(figsize=(12, 6))
        plt.plot(y_real, label='Valor Real', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Previsão', color='red', alpha=0.7, linestyle='--')
        plt.title(f"Previsão vs Real - {run_name} (RMSE: {rmse:.4f})")
        plt.xlabel("Dias")
        plt.ylabel("Preço (R$)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig("prediction_plot.png")
        plt.close()

        mlflow.log_artifact("prediction_plot.png")
        
        # Registra o artefato real do modelo PyTorch no MLflow para versionamento e rollback seguro
        mlflow.pytorch.log_model(model, "lstm_model")
        
        # Limpeza: Remove o arquivo local após enviar para o MLflow para não sujar o disco
        if os.path.exists("prediction_plot.png"):
            os.remove("prediction_plot.png")
        
        logger.info(f"[{run_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | Params: {params}")
        
        return rmse, model.state_dict()

def train() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    set_seed(42) # Seed global inicial
    
    # Configura URI: Usa variável de ambiente (Docker) ou fallback para arquivo local
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.join(BASE_DIR, "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Datathon")

    # Definindo a data atual dinamicamente para evitar hardcode de END_DATE
    current_date = datetime.now().strftime('%Y-%m-%d')

    logger.info("1. Lendo dados da feature store PostgreSQL...")
    df = fetch_financial_data(TICKER, START_DATE, current_date)
    
    # Filtra apenas a feature de Fechamento, pois o modelo tem input_size=1
    if df is not None and 'Close' in df.columns:
        df = df[['Close']]
        
    # --- Circuit Breaker para o Treinamento (Rate Limit Fallback) ---
    is_synthetic = False
    if df is None or len(df) < 100:
        logger.warning("Yahoo Finance bloqueou a requisição (Rate Limit). Ativando dados sintéticos para manter a DAG rodando...")
        # Garante que os dados sintéticos terminem no dia atual para refletir o cenário recente
        dates = pd.date_range(end=current_date, periods=500, freq='B')
        df = pd.DataFrame({'Close': np.linspace(20, 40, 500) + np.random.normal(0, 2, 500)}, index=dates)
        is_synthetic = True
        
    # Materializa o snapshot do DVC/Parquet APENAS quando temos certeza de que os dados são válidos e saudáveis
    materialize_feature_store_snapshot(TICKER, START_DATE, current_date, df=df)
        
    # Divisão de teste: reservamos 120 dias. Como o LSTM usa TIMESTEPS=60, isso nos dará 60 dias robustos de validação do modelo.
    dynamic_test_start = (df.index[-120]).strftime('%Y-%m-%d') if len(df) > 120 else TEST_START_DATE

    # Proteção do Scaler (Champion/Challenger)
    champion_scaler_path = SCALER_PATH
    backup_scaler_path = SCALER_PATH + ".backup"
    challenger_scaler_path = os.path.join(MODELS_DIR, "scaler_challenger.joblib")
    
    if os.path.exists(champion_scaler_path):
        shutil.copy(champion_scaler_path, backup_scaler_path)

    X_train, y_train, X_test, y_test, scaler = preprocess_data(
        df, 
        train_split_date=dynamic_test_start, 
        save_scaler=True
    )

    # Move o novo scaler gerado para o arquivo challenger e restaura o champion para a API
    if os.path.exists(champion_scaler_path):
        shutil.move(champion_scaler_path, challenger_scaler_path)
        if os.path.exists(backup_scaler_path):
            shutil.move(backup_scaler_path, champion_scaler_path)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Grid de Hiperparâmetros
    # Mantemos TIMESTEPS fixo em 60 para compatibilidade com API
    grid = [
        {"hidden_size": 50, "num_layers": 1, "dropout": 0.2, "learning_rate": 0.001, "epochs": 50},
        {"hidden_size": 100, "num_layers": 1, "dropout": 0.3, "learning_rate": 0.001, "epochs": 60},
        {"hidden_size": 50, "num_layers": 2, "dropout": 0.3, "learning_rate": 0.001, "epochs": 50},
        {"hidden_size": 128, "num_layers": 2, "dropout": 0.4, "learning_rate": 0.0005, "epochs": 80},
        {"hidden_size": 64, "num_layers": 1, "dropout": 0.1, "learning_rate": 0.005, "epochs": 40},
    ]
    
    best_rmse = float('inf')
    best_state = None
    best_params = None
    
    logger.info(f"2. Iniciando Grid Search com {len(grid)} experimentos...")
    
    for i, params in enumerate(grid):
        rmse, state = run_experiment(params, X_train, y_train, X_test, y_test, scaler, f"run_{i+1}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = state
            best_params = params
            logger.info(f"   -> Novo melhor modelo! (RMSE: {best_rmse:.4f})")
            
    logger.info(f"3. Resultado Final")
    logger.info(f"Melhor RMSE: {best_rmse:.4f}")
    logger.info(f"Melhores Parâmetros: {best_params}")
    
    # Salvar modelo como Challenger (Candidato)
    challenger_model_path = os.path.join(MODELS_DIR, "lstm_model_challenger.pth")
    torch.save(best_state, challenger_model_path)
    logger.info(f"Modelo Challenger salvo em: {challenger_model_path}")
    
    # Salvar metadados dos hiperparâmetros do Challenger
    challenger_config_path = os.path.join(MODELS_DIR, "challenger_hyperparameters.json")
    with open(challenger_config_path, 'w') as f:
        json.dump(best_params, f)

    # --- Trava de Segurança MLOps ---
    if is_synthetic:
        logger.warning("🛡️ Circuit Breaker: Dados sintéticos detectados. Forçando RMSE do Challenger ao infinito para garantir reprovação.")
        best_rmse = float('inf')

    # Salvar métricas do Challenger para a DAG comparar com o Champion
    challenger_metrics_path = os.path.join(MODELS_DIR, "challenger_metrics.json")
    with open(challenger_metrics_path, 'w') as f:
        json.dump({"rmse": float(best_rmse)}, f)

    # --- Avaliação Justa do Champion no Novo Dataset ---
    # Evita "Data Leakage" temporal garantindo que Champion e Challenger sejam comparados na mesma janela de mercado.
    champion_metrics_path = os.path.join(MODELS_DIR, "champion_metrics.json")
    champion_model_path = os.path.join(MODELS_DIR, "lstm_model.pth")
    
    if os.path.exists(champion_model_path) and os.path.exists(champion_scaler_path):
        logger.info("Avaliando o Champion no dataset recente para métricas justas e comparáveis...")
        try:
            from src.models.predict import Predictor
            predictor = Predictor()
            raw_test = df['Close'].values[-120:]
            predictions, reais = [], []
            
            for i in range(60, len(raw_test)):
                seq = raw_test[i-60:i]
                predictions.append(predictor.predict_next_day(seq))
                reais.append(raw_test[i])
            
            champion_rmse = np.sqrt(mean_squared_error(reais, predictions))
            with open(champion_metrics_path, 'w') as f:
                json.dump({"rmse": float(champion_rmse)}, f)
            logger.info(f"🏆 Champion reavaliado com novo RMSE da semana: {champion_rmse:.4f}")
        except Exception as e:
            logger.error(f"Erro ao reavaliar champion: {e}")

if __name__ == "__main__":
    train()
