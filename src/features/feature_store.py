import os
import pandas as pd
import logging
from src.data.data_loader import fetch_financial_data

logger = logging.getLogger(__name__)

STORE_PATH = "data/processed/feature_store.parquet"

def materialize_feature_store_snapshot(
    ticker: str,
    start_date: str,
    end_date: str,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Materializa um snapshot Parquet a partir da feature store canônica.

    A fonte de verdade dos dados financeiros é o PostgreSQL acessado via
    fetch_financial_data. O Parquet é um artefato derivado para auditoria,
    reprodutibilidade de treino e eventual versionamento com DVC.
    """
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)

    if df is None:
        logger.info("Lendo dados da feature store PostgreSQL para materializar snapshot...")
        df = fetch_financial_data(ticker=ticker, start=start_date, end=end_date)

    if df is None or df.empty:
        logger.warning("Feature store sem dados para materializar snapshot Parquet.")
        return pd.DataFrame()

    df_snapshot = df.copy()
    df_snapshot.index = pd.to_datetime(df_snapshot.index)
    df_snapshot.sort_index(inplace=True)
    df_snapshot.to_parquet(STORE_PATH)
    logger.info("Snapshot Parquet materializado em %s com %d registros.", STORE_PATH, len(df_snapshot))
    return df_snapshot

def update_feature_store(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compatibilidade com chamadas antigas.

    O comportamento atual não baixa dados em paralelo: sincroniza a feature store
    PostgreSQL via fetch_financial_data e gera um snapshot Parquet derivado.
    """
    return materialize_feature_store_snapshot(ticker, start_date, end_date)
