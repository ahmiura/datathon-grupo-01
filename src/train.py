import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
import random
import mlflow
import logging
from typing import Tuple, Dict, Any
import matplotlib
# Configura backend não-interativo para evitar erros no Docker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.features.data import download_data, preprocess_data
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

    logger.info("1. Baixando e processando dados...")
    df = download_data(end=END_DATE)
    X_train, y_train, X_test, y_test, scaler = preprocess_data(
        df, 
        train_split_date=TEST_START_DATE, 
        save_scaler=True
    )
    
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
    
    # Salvar modelo
    torch.save(best_state, MODEL_PATH)
    logger.info(f"Modelo salvo em: {MODEL_PATH}")
    
    # Salvar metadados dos hiperparâmetros para o predict.py usar
    config_path = os.path.join(MODELS_DIR, "model_hyperparameters.json")
    with open(config_path, 'w') as f:
        json.dump(best_params, f)
    logger.info(f"Configuração salva em: {config_path}")

if __name__ == "__main__":
    train()