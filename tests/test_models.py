import torch
from src.models.model import LSTMModel

def test_lstm_model_forward_pass():
    """Garante que a arquitetura da rede neural recebe a entrada e gera a previsão com as dimensões corretas"""
    # Inicializa modelo com parâmetros mínimos para teste rápido
    model = LSTMModel(input_size=1, hidden_size=10, num_layers=1, output_size=1, dropout=0.0)
    model.eval()
    
    # Cria um tensor falso simulando (Batch=2, Sequência=60 dias, Features=1)
    dummy_input = torch.randn(2, 60, 1)
    output = model(dummy_input)
    
    # A saída deve prever 1 valor para cada um dos 2 elementos do batch
    assert output.shape == (2, 1)