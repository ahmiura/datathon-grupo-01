from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.models.predict import Predictor
from src.config import TICKER, MODEL_PATH
from src.serving.schemas import PredictionRequest, PredictionResponse
import uvicorn
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import logging
import time

# Importações para o Agente e LLM (Etapa 2 - Datathon)
from pydantic import BaseModel, Field
from src.agent.react_agent import create_datathon_agent
from src.agent.tools import get_predictor
from src.agent.rag_pipeline import get_cached_vector_store
from langfuse.callback import CallbackHandler
from security.guardrails import check_input, check_output
from src.data.data_loader import fetch_financial_data
from datetime import datetime, timedelta

# Configuração de Logging Estruturado (GAP 09)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictor = None
yf_cache = {"data": [], "timestamp": 0}
CACHE_TTL = 3600  # Tempo de vida do cache em segundos (1 hora)
datathon_agent = None

class ChatRequest(BaseModel):
    message: str = Field(description="Mensagem de texto com a pergunta do usuário para o Agente.")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Qual é a previsão da PETR4 para amanhã e o que diz o relatório sobre os dividendos?"}
            ]
        }
    }

class ChatResponse(BaseModel):
    response: str = Field(description="Resposta processada e validada gerada pelo Agente.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"response": "A previsão do modelo LSTM para o próximo fechamento da PETR4.SA é de R$ 35,12. Além disso, a política de dividendos prevê o pagamento de 45% do lucro líquido ajustado..."}
            ]
        }
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    global datathon_agent
    try:
        if os.path.exists(MODEL_PATH):
            predictor = Predictor()
            logger.info("Modelo carregado com sucesso!")
        else:
            logger.warning("Modelo não encontrado. Execute src/models/train.py primeiro.")

        # Instancia o agente ReAct uma vez para reutilização
        try:
            datathon_agent = create_datathon_agent()
            logger.info("Agente datathon instanciado no lifespan.")
        except Exception as e:
            logger.warning(f"Não foi possível inicializar o agente no lifespan: {e}")

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
    yield

app = FastAPI(
    title="Stock Price Predictor & AI Agent API",
    description="""
Esta API integra um modelo LSTM para prever preços de ações (PETR4) e um Agente ReAct (LLM) capaz de interagir usando dados financeiros, documentos internos via RAG e ferramentas analíticas.

### Recursos Principais:
* **Predição**: Inferência de fechamento para séries financeiras.
* **Agente IA**: Orquestrador LLM integrado a ferramentas externas com proteção de Guardrails (OWASP).
* **MLOps**: Endpoints para verificação de infraestrutura e recarga dinâmica de modelo (Zero Downtime).
""",
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "Equipe Datathon Fase 05"}
)

# Métrica customizada para requisições em andamento
in_progress_gauge = Gauge("http_requests_in_progress", "Requests in progress", ["handler", "method"])

@app.middleware("http")
async def track_in_progress(request: Request, call_next):
    # Incrementa ao receber a requisição
    in_progress_gauge.labels(handler="*", method=request.method).inc()
    try:
        return await call_next(request)
    finally:
        # Decrementa ao finalizar (sucesso ou erro)
        in_progress_gauge.labels(handler="*", method=request.method).dec()

# Configuração de Monitoramento (Prometheus)
Instrumentator().instrument(app).expose(app)


@app.post(
    "/reload-rag",
    summary="Recarregar Banco Vetorial (RAG)",
    description="Limpa o cache em memória do Vector Store (FAISS) forçando a API a reler os PDFs recém-processados pelo Airflow no disco.",
    tags=["Administração e MLOps"]
)
def reload_rag():
    get_cached_vector_store.cache_clear()
    logger.info("Cache do RAG limpo com sucesso. Novos documentos serão lidos na próxima consulta.")
    return {"message": "Cache do RAG (FAISS) invalidado com sucesso."}

@app.post(
    "/predict", 
    response_model=PredictionResponse,
    summary="Realizar previsão de fechamento de ativo (LSTM)",
    description="Gera uma inferência do preço de fechamento para o próximo dia útil. O histórico de 60 dias (`last_60_days`) pode ser fornecido; caso omitido, os últimos dados reais da Feature Store (ou Yahoo Finance via Fallback) serão buscados e inseridos no modelo.",
    response_description="O ticker do ativo analisado e o valor predito.",
    tags=["Modelo Preditivo"]
)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Modelo não carregado ou indisponível.")
    
    try:
        input_data = request.last_60_days

        # Se o usuário não enviou dados, buscamos automaticamente no Yahoo Finance
        if not input_data:
            current_time = time.time()
            # Verifica se o cache está vazio ou se já passou de 1 hora
            if not yf_cache["data"] or (current_time - yf_cache["timestamp"] > CACHE_TTL):
                logger.info(f"Buscando dados recentes para {TICKER} via Feature Store (PostgreSQL)...")
                
                # Busca a partir do data_loader corporativo para aproveitar cache e circuit breakers
                current_date = datetime.now()
                dynamic_start = (current_date - timedelta(days=120)).strftime('%Y-%m-%d')
                df = fetch_financial_data(ticker=TICKER, start=dynamic_start, end=current_date.strftime('%Y-%m-%d'))

                if df.empty or 'Close' not in df.columns:
                    logger.warning("Falha ao retornar dados válidos da Feature Store. Ativando Fallback.")
                    fetched_data = [35.0] * 60
                else:
                    fetched_data = df['Close'].values[-60:].tolist()
                
                if len(fetched_data) < 60:
                    logger.warning(f"Volume insuficiente de dados ({len(fetched_data)} dias). Ativando Circuit Breaker (Fallback).")
                    fetched_data = [35.0] * 60  # Fallback seguro para manter a API 100% online
                
                # Salva no cache
                yf_cache["data"] = fetched_data
                yf_cache["timestamp"] = current_time
                
            input_data = yf_cache["data"]

        price = predictor.predict_next_day(input_data)
        return {"ticker": TICKER, "predicted_price": price}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/reload-model",
    summary="Recarregar modelo (Zero Downtime)",
    description="Lê os pesos (weights) do arquivo do modelo mais recente no disco e recarrega na memória da API, além de limpar o cache da tool usada pelo agente. Essencial para Continuous Training acionado via Airflow.",
    tags=["Administração e MLOps"]
)
def reload_model():
    """Força a recarga do modelo LSTM do disco para a memória."""
    global predictor
    try:
        if os.path.exists(MODEL_PATH):
            predictor = Predictor()
            get_predictor.cache_clear()  # Limpa o cache da Tool usada no /chat
            logger.info("Modelo recarregado com sucesso via endpoint /reload-model!")
            return {"message": "Modelo recarregado com sucesso nas rotas /predict e /chat."}
        else:
            raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado.")
    except Exception as e:
        logger.error(f"Erro ao recarregar modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversar com Agente Financeiro Inteligente (ReAct)",
    description="Envia um prompt para o LLM. O modelo planeja a execução, escolhe as ferramentas (busca em PDF, calculadora, previsão) e retorna a resposta final. Inclui mecanismos de segurança e Guardrails para interceptar PII e Prompt Injection.",
    response_description="Texto final gerado pelo Agente.",
    tags=["Agente de IA"]
)
def chat_with_agent(request: ChatRequest):
    try:
        # 1. Guardrail de Input (Segurança Etapa 4)
        is_safe_input, error_msg = check_input(request.message)
        if not is_safe_input:
            raise HTTPException(status_code=403, detail=error_msg)
            
        langfuse_handler = CallbackHandler()
        # Reutiliza agente instanciado no lifespan quando disponível
        agent_executor = datathon_agent if datathon_agent is not None else create_datathon_agent()
        result = agent_executor.invoke({"input": request.message}, config={"callbacks": [langfuse_handler]})
        
        # 2. Guardrail de Output (Segurança Etapa 4)
        is_safe_output, safe_output = check_output(result["output"])
        
        return {"response": safe_output}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erro na execução do agente LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    summary="Health Check (Liveness Probe)",
    description="Retorna se a API está em execução no contêiner e reporta internamente se o modelo preditivo foi inicializado na memória.",
    tags=["Monitoramento"]
)
def health():
    return {"status": "ok", "model_loaded": bool(predictor)}

@app.get(
    "/ready",
    summary="Readiness Probe",
    description="Valida se a aplicação está apta a receber tráfego e requisições de clientes. Retorna um erro HTTP 503 se o modelo não estiver ativamente servido.",
    tags=["Monitoramento"]
)
def ready():
    if predictor:
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Modelo não carregado")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)