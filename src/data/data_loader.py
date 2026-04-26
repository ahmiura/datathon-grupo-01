import os
import logging
import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from datetime import date, timedelta
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variáveis globais para gerenciar o Rate Limit do Yahoo (Anti-Spam)
LAST_YF_BLOCK_TIME = 0
# Parâmetros configuráveis via env
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", 3600))  # Cooldown padrão 1 hora
YF_RETRIES = int(os.getenv("YF_RETRIES", 3))
YF_BACKOFF_FACTOR = float(os.getenv("YF_BACKOFF_FACTOR", 2.0))
YF_TIMEOUT = int(os.getenv("YF_TIMEOUT", 30))
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", 5))

def get_db_connection():
    """Cria conexão com o PostgreSQL da feature store/cache."""
    return psycopg2.connect(
        host=os.getenv("FEATURE_DB_HOST", os.getenv("DB_HOST", "localhost")),
        port=os.getenv("FEATURE_DB_PORT", os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("FEATURE_DB_NAME", os.getenv("DB_NAME", "postgres")),
        user=os.getenv("FEATURE_DB_USER", os.getenv("DB_USER", "postgres")),
        password=os.getenv("FEATURE_DB_PASS", os.getenv("DB_PASS", "postgres")),
        connect_timeout=DB_CONNECT_TIMEOUT
    )

def fetch_financial_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Coleta dados financeiros, utilizando a feature store PostgreSQL como cache.
    Se os dados não estiverem no cache, busca no Yahoo Finance via yfinance (com retries/backoff) e os armazena de forma resiliente.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker VARCHAR(20),
                date DATE,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume BIGINT,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()

        # Tenta buscar os dados do cache primeiro
        query = """
            SELECT date, open, high, low, close, volume 
            FROM stock_prices 
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date ASC
        """
        cursor.execute(query, (ticker, start, end))
        rows = cursor.fetchall()
        
        # Heurística para detectar se o cache está incompleto ou desatualizado
        end_date = pd.to_datetime(end)
        last_cached_date = pd.to_datetime(rows[-1][0]) if rows else pd.to_datetime('1970-01-01')
        is_cache_stale = (end_date.date() - last_cached_date.date()).days > 5
        
        is_cache_incomplete = not rows or len(rows) < (end_date - pd.to_datetime(start)).days * 0.5 or is_cache_stale

        if is_cache_incomplete:
            global LAST_YF_BLOCK_TIME
            if time.time() - LAST_YF_BLOCK_TIME < COOLDOWN_SECONDS:
                logger.warning(f"Yahoo Finance em Cooldown (Rate Limit). Usando cache local para {ticker}.")
                df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                if not df.empty:
                    df.set_index('Date', inplace=True)
                return df

            logger.info(f"Cache para {ticker} ({start} a {end}) incompleto/desatualizado. Consultando Yahoo Finance...")

            # Tenta baixar do Yahoo com retries e backoff exponencial
            attempt = 0
            df_yf = pd.DataFrame()
            while attempt < YF_RETRIES:
                try:
                    df_yf = yf.download(ticker, start=start, end=end, progress=False, timeout=YF_TIMEOUT)
                    # Se conseguiu sem exceção, sai do loop
                    break
                except Exception as ex:
                    attempt += 1
                    backoff = YF_BACKOFF_FACTOR ** attempt
                    logger.warning(f"yfinance download failed (attempt {attempt}/{YF_RETRIES}): {ex}. Backing off {backoff}s")
                    time.sleep(backoff)

            # Se df_yf continua vazio, vamos usar fallback abaixo
            
            # Corrige o formato MultiIndex retornado pelas versões novas do yfinance
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            if not df_yf.empty:
                df_yf_reset = df_yf.reset_index()
                records = [
                    (ticker, row['Date'].date(), float(row['Open']), float(row['High']), 
                     float(row['Low']), float(row['Close']), int(row['Volume']) if pd.notna(row['Volume']) else 0)
                    for _, row in df_yf_reset.iterrows()
                ]
                
                insert_query = """
                    INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (ticker, date) DO NOTHING
                """
                execute_values(cursor, insert_query, records)
                conn.commit()
                logger.info(f"✅ Inseridos/Verificados {len(records)} registros no PostgreSQL para {ticker}.")
                df = df_yf
            else:
                logger.warning(f"Falha ao buscar dados no Yahoo Finance para {ticker}. Usando dados de cache existentes.")
                LAST_YF_BLOCK_TIME = time.time()  # Ativa o cooldown para não metralhar a API
                df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                if not df.empty:
                    df.set_index('Date', inplace=True)
        else:
            logger.info(f"⚡ Dados de {ticker} ({start} a {end}) carregados do Cache (PostgreSQL).")
            df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)

        if not df.empty:
            df.index = pd.to_datetime(df.index)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    except (psycopg2.OperationalError, Exception) as e:
        logger.error(f"Erro no data_loader para {ticker}: {e}. Tentando fallback para yfinance direto.")
        # fallback: tentar yfinance com retries
        attempt = 0
        while attempt < YF_RETRIES:
            try:
                df_fallback = yf.download(ticker, start=start, end=end, progress=False, timeout=YF_TIMEOUT)
                if isinstance(df_fallback.columns, pd.MultiIndex):
                    df_fallback.columns = df_fallback.columns.get_level_values(0)
                return df_fallback
            except Exception as yf_e:
                attempt += 1
                backoff = YF_BACKOFF_FACTOR ** attempt
                logger.warning(f"Fallback yfinance failed (attempt {attempt}/{YF_RETRIES}): {yf_e}. Backoff {backoff}s")
                time.sleep(backoff)
        logger.error(f"Fallback para yfinance também falhou após {YF_RETRIES} tentativas.")
        return pd.DataFrame()
    finally:
        if conn:
            cursor.close()
            conn.close()
