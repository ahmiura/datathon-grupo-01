import os
import pandas as pd
import yfinance as yf
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

STORE_PATH = "data/processed/feature_store.parquet"

def update_feature_store(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Atualiza a Feature Store de forma INCREMENTAL (Resolve GAP 03).
    Evita o padrão destrutivo (Full-Flush) baixando apenas os dados faltantes (Deltas).
    """
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    
    if os.path.exists(STORE_PATH):
        logger.info("Feature Store local encontrada. Lendo dados...")
        df_existente = pd.read_parquet(STORE_PATH)
        
        if not df_existente.empty:
            ultima_data = df_existente.index.max()
            logger.info(f"Última data na store: {ultima_data.date()}. Buscando deltas...")
            
            start_delta = (ultima_data + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Só busca na API externa se o delta (próximo dia) for menor que a data final requerida
            if start_delta < end_date:
                df_novo = yf.download(ticker, start=start_delta, end=end_date, progress=False)
                if not df_novo.empty:
                    df_final = pd.concat([df_existente, df_novo])
                    df_final.to_parquet(STORE_PATH)
                    logger.info(f"Feature Store atualizada incrementalmente (+{len(df_novo)} registros).")
                    return df_final
            
            logger.info("Feature Store já está totalmente atualizada.")
            return df_existente

    # Carga Inicial (Bulk Load) se o banco de features não existir
    logger.info("Feature Store não encontrada ou vazia. Realizando carga inicial...")
    df_inicial = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df_inicial.to_parquet(STORE_PATH)
    return df_inicial