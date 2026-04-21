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
from langfuse.callback import CallbackHandler
from security.guardrails import check_input, check_output

# Configuração de Logging Estruturado (GAP 09)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictor = None
yf_cache = {"data": [], "timestamp": 0}
CACHE_TTL = 3600  # Tempo de vida do cache em segundos (1 hora)

class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        if os.path.exists(MODEL_PATH):
            predictor = Predictor()
            logger.info("Modelo carregado com sucesso!")
        else:
            logger.warning("Modelo não encontrado. Execute src/train.py primeiro.")
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

@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    try:
        # 1. Guardrail de Input (Segurança Etapa 4)
        is_safe_input, error_msg = check_input(request.message)
        if not is_safe_input:
            raise HTTPException(status_code=403, detail=error_msg)
            
        langfuse_handler = CallbackHandler()
        agent_executor = create_datathon_agent()
        result = agent_executor.invoke({"input": request.message}, config={"callbacks": [langfuse_handler]})
        
        # 2. Guardrail de Output (Segurança Etapa 4)
        is_safe_output, safe_output = check_output(result["output"])
        
        return {"response": safe_output}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erro na execução do agente LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)