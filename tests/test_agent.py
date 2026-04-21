from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from src.agent.tools import calculator, get_current_stock_price, search_company_documents, predict_petr4_close_price

def test_calculator_tool_valid():
    """Garante que a ferramenta matemática do Agente calcula corretamente"""
    result = calculator.invoke("2 * 21.5")
    assert result == "43.0"

def test_calculator_tool_invalid_security():
    """Garante que a ferramenta bloqueia tentativas de injeção de código"""
    # A calculadora só permite números e operadores matemáticos básicos
    result = calculator.invoke("import os; os.system('rm -rf /')")
    assert "Erro" in result

def test_get_current_stock_price_tool():
    """Garante que a ferramenta de busca de ações lida com a requisição ao Yahoo Finance"""
    result = get_current_stock_price.invoke("PETR4.SA")
    assert isinstance(result, str)

def test_search_company_documents_tool():
    """Garante que a ferramenta de RAG consegue ser invocada sem quebrar a aplicação"""
    result = search_company_documents.invoke("Dividendos")
    assert isinstance(result, str)

@patch("src.agent.tools.fetch_financial_data")
@patch("src.agent.tools.get_predictor")
def test_predict_petr4_close_price_tool(mock_get_predictor, mock_fetch_financial_data):
    """Testa a ferramenta de integração com o LSTM simulando a feature store e o modelo"""
    mock_predictor = MagicMock()
    mock_predictor.predict_next_day.return_value = 35.5
    mock_get_predictor.return_value = mock_predictor
    
    mock_fetch_financial_data.return_value = pd.DataFrame({"Close": np.array([30.0] * 60)})
    
    result = predict_petr4_close_price.invoke("")
    assert "35.5" in result

@patch("src.agent.react_agent.ChatGoogleGenerativeAI")
def test_create_datathon_agent(mock_llm):
    """Garante que o cérebro do agente é instanciado corretamente com todas as ferramentas"""
    from src.agent.react_agent import create_datathon_agent
    agent = create_datathon_agent()
    assert agent is not None

@patch("src.agent.rag_pipeline.FAISS")
@patch("src.agent.rag_pipeline.os.path.exists", return_value=True)
def test_rag_pipeline_query(mock_exists, mock_faiss):
    """Testa a recuperação de documentos no RAG sem precisar de PDFs reais"""
    from src.agent.rag_pipeline import query_documents
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Texto confidencial do PDF da empresa."
    mock_vs.similarity_search.return_value = [mock_doc]
    mock_faiss.load_local.return_value = mock_vs
    
    result = query_documents("Teste de RAG")
    assert "Texto confidencial" in result
