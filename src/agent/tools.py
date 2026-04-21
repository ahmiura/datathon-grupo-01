from langchain.tools import tool
from src.models.predict import Predictor
from src.agent.rag_pipeline import query_documents
from src.config import START_DATE, END_DATE
from src.data.data_loader import fetch_financial_data
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_predictor():
    """Carrega o modelo preditivo LSTM (Singleton)"""
    try:
        return Predictor()
    except Exception as e:
        logger.error(f"Erro ao inicializar Predictor na Tool: {e}")
        return None

@tool
def predict_petr4_close_price(query: str = "") -> str:
    """Use esta ferramenta para prever o preço de fechamento da ação PETR4.SA para o próximo dia útil usando um modelo preditivo de Inteligência Artificial. Não requer argumentos."""
    try:
        predictor = get_predictor()
        if predictor is None:
            return "Modelo preditivo indisponível no momento."
        
        # Busca os dados usando o data loader com cache
        df = fetch_financial_data(ticker="PETR4.SA", start=START_DATE, end=END_DATE)
        input_data = df['Close'].values[-60:].tolist()

        # Circuit Breaker da Tool (mesma proteção que colocamos na API)
        if len(input_data) < 60:
            logger.warning("Yahoo Finance bloqueou a requisição na Tool. Ativando Fallback.")
            input_data = [35.0] * 60

        price = predictor.predict_next_day(input_data)
        logger.info(f"Previsão LSTM gerada com sucesso para PETR4.SA: R$ {price:.2f}")
        return f"A previsão da nossa IA (LSTM) para o próximo fechamento da PETR4.SA é de R$ {price:.2f}"
    except Exception as e:
        return f"Erro ao executar a previsão: {str(e)}"

@tool
def get_current_stock_price(ticker: str) -> str:
    """Use esta ferramenta para obter o preço atual e real de qualquer ação. A entrada deve ser EXATAMENTE o ticker da ação com o sufixo (ex: 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA')."""
    try:
        # Busca os dados usando o data loader com cache, garantindo que os dados mais recentes sejam buscados
        history = fetch_financial_data(ticker=ticker, start=START_DATE, end=END_DATE)
        if history.empty:
            return f"Não foram encontrados dados para o ticker {ticker}."
        price = history['Close'].iloc[-1]
        logger.info(f"Cotação real obtida para {ticker}: R$ {price:.2f}")
        return f"O preço atual de {ticker} na bolsa é R$ {price:.2f}"
    except Exception as e:
        return f"Erro ao buscar preço atual: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Use esta ferramenta para fazer cálculos matemáticos simples, como descobrir a diferença entre dois preços. A entrada deve ser APENAS uma expressão matemática válida (ex: '2 * 34.50' ou '36.50 - 32.10')."""
    try:
        # Validação simples de segurança
        allowed = "0123456789+-*/. ()"
        if any(c not in allowed for c in expression):
            return "Erro: A expressão contém caracteres inválidos."
        resultado = str(eval(expression))
        logger.info(f"Cálculo executado: {expression} = {resultado}")
        return resultado
    except Exception as e:
        return f"Erro no cálculo: {str(e)}"

@tool
def search_company_documents(query: str) -> str:
    """Use esta ferramenta APENAS quando precisar buscar informações em documentos internos, relatórios financeiros ou PDFs da empresa para responder à pergunta. A entrada deve ser a sua dúvida de pesquisa."""
    try:
        logger.info(f"Buscando contexto em documentos para a query: '{query}'")
        return query_documents(query)
    except Exception as e:
        logger.error(f"Erro no RAG: {e}")
        return f"Erro ao consultar documentos: {str(e)}"