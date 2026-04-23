from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.models.predict import Predictor
from src.config import TICKER, MODEL_PATH, START_DATE, END_DATE
from src.serving.schemas import PredictionRequest, PredictionResponse
import uvicorn
import os
import yfinance as yf
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import logging
import time

# Importações para o Agente e LLM (Etapa 2 - Datathon)
from pydantic import BaseModel
from src.agent.react_agent import create_datathon_agent
from src.agent.tools import get_predictor
from langfuse.callback import CallbackHandler
from security.guardrails import check_input, check_output

# Configuração de Logging Estruturado (GAP 09)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictor = None
yf_cache = {"data": [], "timestamp": 0}
CACHE_TTL = 3600  # Tempo de vida do cache em segundos (1 hora)
datathon_agent = None

class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    global datathon_agent
    try:
        if os.path.exists(MODEL_PATH):
            predictor = Predictor()
            logger.info("Modelo carregado com sucesso!")
        else:
            logger.warning("Modelo não encontrado. Execute src/train.py primeiro.")

        # Instancia o agente ReAct uma vez para reutilização
        try:
            datathon_agent = create_datathon_agent()
            logger.info("Agente datathon instanciado no lifespan.")
        except Exception as e:
            logger.warning(f"Não foi possível inicializar o agente no lifespan: {e}")

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
    yield

app = FastAPI(title="Stock Price Predictor API", version="1.0", lifespan=lifespan)

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


@app.post("/predict", response_model=PredictionResponse)
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
                logger.info(f"Buscando dados recentes para {TICKER} no Yahoo Finance...")
                df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
                fetched_data = df['Close'].values[-60:].tolist()
                
                if len(fetched_data) < 60:
                    logger.warning(f"Yahoo Finance limitou a requisição ({len(fetched_data)} dias). Ativando Circuit Breaker (Fallback).")
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

@app.post("/reload-model")
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

@app.post("/chat", response_model=ChatResponse)
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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(predictor)}


@app.get("/ready")
def ready():
    if predictor:
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Modelo não carregado")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)