import os
from unittest.mock import patch, MagicMock

os.environ["GOOGLE_API_KEY"] = "fake-key"
from src.agent.rag_pipeline import build_or_load_vector_store, update_vector_store

@patch("src.agent.rag_pipeline.os.path.exists")
@patch("src.agent.rag_pipeline.FAISS")
@patch("src.agent.rag_pipeline.GoogleGenerativeAIEmbeddings")
def test_build_or_load_vector_store_existing(mock_embed, mock_faiss, mock_exists):
    """Garante carregamento de banco vetorial já existente"""
    mock_exists.return_value = True
    mock_faiss.load_local.return_value = "mocked_store"
    
    vs = build_or_load_vector_store()
    assert vs == "mocked_store"

@patch("src.agent.rag_pipeline.os.path.exists")
def test_build_or_load_vector_store_no_path(mock_exists):
    """Garante tratamento correto quando PDF não existe"""
    mock_exists.return_value = False
    vs = build_or_load_vector_store()
    assert vs is None

@patch("src.agent.rag_pipeline.os.path.exists", return_value=True)
@patch("src.agent.rag_pipeline.PyPDFDirectoryLoader")
@patch("src.agent.rag_pipeline.FAISS")
@patch("src.agent.rag_pipeline.GoogleGenerativeAIEmbeddings")
def test_update_vector_store(mock_embed, mock_faiss, mock_loader, mock_exists):
    """Garante que a reconstrução do Vector Store (RAG) execute via testes"""
    mock_instance = MagicMock()
    mock_instance.load.return_value = [MagicMock(page_content="page 1")]
    mock_loader.return_value = mock_instance
    
    mock_faiss.from_documents.return_value = MagicMock()
    
    result = update_vector_store()
    assert result is True