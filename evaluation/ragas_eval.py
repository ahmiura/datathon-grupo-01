import json
import os
import logging
import time
import asyncio
import mlflow
import requests
from src.agent.rag_pipeline import query_documents
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trava global para garantir que requisições async não atropelem umas às outras
gemini_lock = None

# Wrapper para resolver conflito de kwargs entre Ragas e a API do Google
class SafeGemini(ChatGoogleGenerativeAI):
    def generate(self, *args, **kwargs):
        kwargs.pop('temperature', None)
        time.sleep(4)  # Pausa síncrona
        return super().generate(*args, **kwargs)
        
    async def agenerate(self, *args, **kwargs):
        kwargs.pop('temperature', None)
        global gemini_lock
        if gemini_lock is None:
            gemini_lock = asyncio.Lock()
            
        async with gemini_lock:  # Fila indiana estrita
            await asyncio.sleep(4)  # Espera 4 segundos SOZINHO na fila
            return await super().agenerate(*args, **kwargs)

def load_golden_set(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation(golden_set_path: str, max_samples: int = None):
    logger.info("Carregando Golden Set...")
    golden_data = load_golden_set(golden_set_path)
    
    if max_samples:
        logger.info(f"Limitando avaliação RAGAS a {max_samples} amostras (Modo Dev/Teste Rápido)...")
        golden_data = golden_data[:max_samples]
    
    logger.info("Gerando respostas frescas do Agente via API para avaliação contínua real...")
    api_url = os.getenv("API_URL", "http://api:8000/chat")
    live_data = []
    
    for idx, item in enumerate(golden_data):
        query = item["query"]
        logger.info(f"[{idx+1}/{len(golden_data)}] Obtendo inferência para: '{query}'")
        
        # 1. Obtém o contexto real que o RAG recuperaria hoje
        real_context = query_documents(query)
        item["contexts"] = [real_context] if real_context else ["Sem contexto retornado."]
        
        # 2. Bate na API para pegar a resposta real do Agente com o modelo atualizado
        try:
            resp = requests.post(api_url, json={"message": query}, timeout=60)
            item["answer"] = resp.json().get("response", "Erro na resposta") if resp.status_code == 200 else f"Erro API: {resp.status_code}"
        except Exception as e:
            logger.error(f"Erro ao comunicar com a API: {e}")
            item["answer"] = "Falha de Conexão com o Agente."
            
        live_data.append(item)

    # Salva os resultados gerados ao vivo para que o LLM Judge (Business Metrics) avalie exatamente as mesmas respostas
    latest_eval_path = "data/golden_set/latest_eval_predictions.json"
    with open(latest_eval_path, "w", encoding="utf-8") as f:
        json.dump(live_data, f, ensure_ascii=False, indent=2)

    data_for_ragas = {
        "question": [item["query"] for item in live_data],
        "answer": [item["answer"] for item in live_data],
        "contexts": [item["contexts"] for item in live_data],
        "ground_truth": [item["expected_answer"] for item in live_data]
    }
    
    dataset = Dataset.from_dict(data_for_ragas)
    
    logger.info("Configurando LLM e Embeddings do Gemini para atuar como Juiz (LLM-as-a-judge)...")
    # Usamos o Gemini para avaliar a qualidade
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemma-3-27b-it")
    # Usamos o nosso Wrapper com a temperatura fixada de forma segura na base
    evaluator_llm = SafeGemini(model=model_name, temperature=0.0)
    embed_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model=embed_model)
    
    logger.info("Iniciando avaliação RAGAS (Isso pode levar alguns minutos dependendo da API)...")
    
    # As 4 métricas exigidas pelo GAP e Etapa 3 do Requisito
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    # Configuração Anti-Rate Limit (Contas FREE limitadas a 15 RPM)
    # max_workers=1 impede que o Ragas dispare dezenas de requisições simultâneas
    # e entre em um loop de backoff exponencial infinito.
    run_config = RunConfig(
        max_workers=1, 
        max_retries=10
    )
    
    results = evaluate(dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings, run_config=run_config, raise_exceptions=False)
    
    logger.info("Avaliação concluída!")
    logger.info(f"Resultados Consolidados: \n{results}")
    
    logger.info("Enviando métricas para o MLflow...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Datathon_LLM_Evaluation")
    
    with mlflow.start_run(run_name="RAGAS_Technical_Eval"):
        mlflow.log_param("max_samples", max_samples if max_samples else "All_Golden_Set")
        mlflow.log_metrics({
            "ragas_faithfulness": results.get("faithfulness", 0.0),
            "ragas_answer_relevancy": results.get("answer_relevancy", 0.0),
            "ragas_context_precision": results.get("context_precision", 0.0),
            "ragas_context_recall": results.get("context_recall", 0.0)
        })
        
    return results

if __name__ == "__main__":
    # max_samples=3 devido os limites do LLM do Google (gratuito)
    run_evaluation("data/golden_set/golden_set.json", max_samples=3)
