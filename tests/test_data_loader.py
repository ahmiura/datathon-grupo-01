import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.data_loader import fetch_financial_data

@patch("src.data.data_loader.get_db_connection")
@patch("src.data.data_loader.yf.download")
def test_fetch_financial_data_cache_miss(mock_yf, mock_db):
    """Testa o data loader simulando falta de cache (busca no Yahoo Finance)"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor
    mock_db.return_value = mock_conn

    df_mock = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01"]),
        "Open": [10.0],
        "High": [11.0],
        "Low": [9.0],
        "Close": [10.5],
        "Volume": [1000]
    })
    df_mock.set_index("Date", inplace=True)
    mock_yf.return_value = df_mock

    df = fetch_financial_data("PETR4.SA", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert "Close" in df.columns

@patch("src.data.data_loader.get_db_connection")
def test_fetch_financial_data_cache_hit(mock_db):
    """Testa o data loader retornando dados direto do PostgreSQL"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_cursor.fetchall.return_value = [
        (pd.to_datetime("2024-05-01").date(), 10.0, 11.0, 9.0, 10.5, 1000)
    ]
    mock_conn.cursor.return_value = mock_cursor
    mock_db.return_value = mock_conn

    df = fetch_financial_data("PETR4.SA", "2024-05-01", "2024-05-01")
    assert not df.empty
    assert len(df) == 1