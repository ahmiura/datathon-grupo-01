import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.predict import Predictor

@patch("src.models.predict.torch.load")
@patch("src.models.predict.joblib.load")
@patch("src.models.predict.os.path.exists", return_value=False)
def test_predictor_initialization_and_prediction(mock_exists, mock_joblib, mock_torch):
    """Garante que a classe Predictor inicializa e faz a previsão corretamente (usando mocks)"""
    
    # Gera um dicionário de pesos válido para enganar o PyTorch no teste
    from src.models.model import LSTMModel
    dummy_model = LSTMModel(hidden_size=50, num_layers=1, dropout=0.2)
    mock_torch.return_value = dummy_model.state_dict()
    
    # 1. Configura os mocks do Scaler para não precisar de arquivos reais no disco
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((60, 1))
    mock_scaler.inverse_transform.return_value = np.array([[35.5]])
    mock_joblib.return_value = mock_scaler
    
    predictor = Predictor()
    
    # 2. Configura o mock do forward pass do modelo PyTorch (LSTM)
    mock_output = MagicMock()
    mock_output.cpu.return_value.numpy.return_value = np.array([[35.5]])
    
    predictor.model = MagicMock(return_value=mock_output)
    
    # 3. Testa o método de previsão
    price = predictor.predict_next_day([30.0] * 60)
    assert price == 35.5

@patch("src.models.predict.torch.load")
@patch("src.models.predict.joblib.load")
@patch("src.models.predict.os.path.exists", return_value=True)
@patch("builtins.open")
@patch("src.models.predict.json.load")
def test_predictor_dynamic_config(mock_json, mock_open, mock_exists, mock_joblib, mock_torch):
    """Garante que o Predictor carrega a configuração dinâmica do JSON se ela existir no disco"""
    mock_json.return_value = {"hidden_size": 64, "num_layers": 2, "dropout": 0.1}
    
    from src.models.model import LSTMModel
    dummy_model = LSTMModel(hidden_size=64, num_layers=2, dropout=0.1)
    mock_torch.return_value = dummy_model.state_dict()
    
    predictor = Predictor()
    assert predictor.model.hidden_size == 64

@patch("src.models.predict.torch.load")
@patch("src.models.predict.joblib.load")
@patch("src.models.predict.os.path.exists", return_value=True)
@patch("builtins.open", side_effect=Exception("Erro simulado"))
def test_predictor_dynamic_config_error(mock_open, mock_exists, mock_joblib, mock_torch):
    """Garante o fallback para os valores defaults se o JSON der erro ou estiver corrompido"""
    from src.models.model import LSTMModel
    mock_torch.return_value = LSTMModel().state_dict()
    predictor = Predictor()
    assert predictor.model is not None