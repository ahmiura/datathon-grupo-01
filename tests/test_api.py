from fastapi.testclient import TestClient
from unittest.mock import patch
import os

os.environ["GOOGLE_API_KEY"] = "fake-api-key"
from src.serving.app import app

client = TestClient(app)

def test_docs_health_check():
    """Garante que a API está de pé e a documentação está acessível"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_metrics_endpoint():
    """Garante que o endpoint de métricas do Prometheus está acessível"""
    response = client.get("/metrics")
    assert response.status_code == 200

def test_predict_endpoint():
    """Testa a rota de previsão com o fallback automático (sem dados no body)"""
    response = client.post("/predict", json={})
    assert response.status_code in [200, 503] # 200 se o modelo estiver treinado, 503 se faltar o arquivo .pth

@patch("src.serving.app.create_datathon_agent")
def test_chat_endpoint(mock_create_agent):
    """Garante que a rota de chat processa mensagens simulando a resposta do Agente"""
    # Configura o mock para evitar chamadas reais à API do Google durante os testes
    mock_agent = mock_create_agent.return_value
    mock_agent.invoke.return_value = {"output": "Esta é uma resposta simulada do LLM."}
    
    response = client.post("/chat", json={"message": "Qual a previsão da PETR4?"})
    assert response.status_code == 200
    assert response.json()["response"] == "Esta é uma resposta simulada do LLM."

@patch("src.serving.app.predictor")
def test_predict_endpoint_internal_error(mock_predictor):
    """Garante que erros internos no modelo de previsão retornam status 500"""
    # Força a API a falhar ao tentar prever
    mock_predictor.predict_next_day.side_effect = Exception("Erro Simulado no PyTorch")
    response = client.post("/predict", json={"last_60_days": [30.0] * 60})
    assert response.status_code == 500

@patch("src.serving.app.create_datathon_agent")
def test_chat_endpoint_internal_error(mock_create_agent):
    """Garante que falhas na API do LLM retornam status 500 para o usuário"""
    # Força a API a falhar ao tentar chamar o Agente
    mock_create_agent.side_effect = Exception("Google GenAI Offline")
    response = client.post("/chat", json={"message": "Oi"})
    assert response.status_code == 500

@patch("src.serving.app.Predictor")
@patch("src.serving.app.os.path.exists", return_value=True)
@patch("src.serving.app.get_predictor")
def test_reload_model_endpoint(mock_get_predictor, mock_exists, mock_predictor):
    """Garante que a rota de recarga do modelo limpa o cache e reinicializa o Predictor"""
    response = client.post("/reload-model")
    assert response.status_code == 200
    assert "recarregado com sucesso" in response.json()["message"]
    mock_predictor.assert_called_once()
    mock_get_predictor.cache_clear.assert_called_once()