import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.train import run_experiment, train

@patch("src.train.mlflow")
@patch("src.train.os.remove")
@patch("src.train.os.path.exists", return_value=True)
@patch("src.train.plt.savefig")
def test_run_experiment(mock_savefig, mock_exists, mock_remove, mock_mlflow):
    """Testa a execução de um único experimento de treinamento (1 epoch)"""
    X_train = torch.randn(10, 60, 1)
    y_train = torch.randn(10, 1)
    
    mock_scaler = MagicMock()
    mock_scaler.inverse_transform.side_effect = lambda x: x
    
    params = {
        "hidden_size": 10,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.1,
        "epochs": 1
    }
    
    rmse, state_dict = run_experiment(params, X_train, y_train, X_train, y_train, mock_scaler, "test_run")
    assert rmse >= 0
    assert isinstance(state_dict, dict)
    mock_mlflow.log_artifact.assert_any_call("data/processed/feature_store.parquet", artifact_path="data")

@patch("src.train.run_experiment")
@patch("src.train.preprocess_data")
@patch("src.train.materialize_feature_store_snapshot")
@patch("src.train.fetch_financial_data")
@patch("src.train.torch.save")
@patch("src.train.json.dump")
@patch("builtins.open")
@patch("src.train.mlflow")
def test_train_function(mock_mlflow, mock_open, mock_json, mock_save, mock_fetch, mock_snapshot, mock_preprocess, mock_run):
    """Testa a orquestração do grid search no train.py raiz"""
    mock_fetch.return_value = pd.DataFrame({"Close": [10.0]*100})
    mock_snapshot.return_value = pd.DataFrame({"Close": [10.0]*100})
    mock_preprocess.return_value = (np.random.randn(10, 60, 1), np.random.randn(10, 1), np.random.randn(5, 60, 1), np.random.randn(5, 1), MagicMock())
    mock_run.return_value = (1.5, {"weight": 1})
    
    train()
    mock_save.assert_called_once()

@patch("src.models.train.mlflow")
@patch("src.models.train.os.remove")
@patch("src.models.train.os.path.exists", return_value=True)
@patch("src.models.train.plt.savefig")
def test_models_run_experiment_logs_snapshot_artifact(mock_savefig, mock_exists, mock_remove, mock_mlflow):
    """Garante que o treino versiona o snapshot Parquet como artifact no MLflow"""
    from src.models.train import run_experiment as run_models_experiment

    X_train = torch.randn(10, 60, 1)
    y_train = torch.randn(10, 1)

    mock_scaler = MagicMock()
    mock_scaler.inverse_transform.side_effect = lambda x: x

    params = {
        "hidden_size": 10,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.1,
        "epochs": 1
    }

    rmse, state_dict = run_models_experiment(params, X_train, y_train, X_train, y_train, mock_scaler, "test_run")
    assert rmse >= 0
    assert isinstance(state_dict, dict)
    mock_mlflow.log_artifact.assert_any_call("data/processed/feature_store.parquet", artifact_path="data")

@patch("src.models.train.run_experiment")
@patch("src.models.train.preprocess_data")
@patch("src.models.train.materialize_feature_store_snapshot")
@patch("src.models.train.fetch_financial_data")
@patch("src.models.train.torch.save")
@patch("src.models.train.json.dump")
@patch("builtins.open")
@patch("src.models.train.mlflow")
def test_train_models_function(mock_mlflow, mock_open, mock_json, mock_save, mock_fetch, mock_snapshot, mock_preprocess, mock_run):
    """Testa a orquestração do grid search no src/models/train.py"""
    mock_fetch.return_value = pd.DataFrame({"Close": [10.0]*100})
    mock_snapshot.return_value = pd.DataFrame({"Close": [10.0]*100})
    mock_preprocess.return_value = (np.random.randn(10, 60, 1), np.random.randn(10, 1), np.random.randn(5, 60, 1), np.random.randn(5, 1), MagicMock())
    mock_run.return_value = (1.5, {"weight": 1})
    
    from src.models.train import train as train_models
    train_models()
    mock_save.assert_called_once()
