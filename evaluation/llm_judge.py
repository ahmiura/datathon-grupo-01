import json
import os
import time
import logging
import warnings
import mlflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Suprime warnings de depreciação do Google e Cryptography para limpar os logs
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prompt para o LLM-as-a-judge
JUDGE_PROMPT = PromptTemplate.from_template("""Você é um auditor de conformidade e qualidade em uma instituição financeira.
Sua tarefa é avaliar a resposta dada por um Agente de Inteligência Artificial a uma pergunta de um usuário.

Pergunta do Usuário: {query}
Resposta do Agente: {answer}

Avalie a resposta do Agente atribuindo uma nota inteira de 1 a 5 para os seguintes 3 critérios:
1. Tom de Voz Corporativo: A linguagem é profissional, educada e adequada para o mercado financeiro? (1=Gírias/Inadequado, 5=Extremamente profissional)
2. Concisão: A resposta é direta ao ponto, sem enrolação ou informações irrelevantes? (1=Prolixa/Confusa, 5=Direta/Clara)
3. Alinhamento à Política de Risco (Negócio): A resposta evita fazer promessas absolutas de lucro, garantias irreais ou recomendações irresponsáveis de compra/venda? (1=Faz promessas/Irresponsável, 5=Cautelosa/Informativa/Neutra)

Retorne APENAS um objeto JSON válido no seguinte formato exato, sem formatação markdown extra e sem blocos de código:
{{
    "tom_corporativo": <nota>,
    "concisao": <nota>,
    "alinhamento_risco": <nota>,
    "justificativa": "<breve justificativa>"
}}
""")

def run_llm_judge(golden_set_path: str, max_samples: int = 3):
    """Executa a avaliação LLM-as-a-judge sobre as respostas geradas no golden set."""
    
    logger.info("⏳ Aguardando 20 segundos para esfriar o Rate Limit da API após o RAGAS...")
    time.sleep(20)
    
    logger.info("⚖️ Iniciando avaliação LLM-as-a-judge (Critérios de Negócio)...")
    
    with open(golden_set_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Limita a amostra padrão para não estourar o Rate Limit de contas gratuitas
    # Em um ambiente corporativo pago, isso rodaria no dataset completo
    sample_data = data[:max_samples]
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    # Temperatura 0.0 é fundamental para o Juiz ser determinístico e justo
    # max_retries evita que o script trave por muito tempo se sofrer block
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, max_retries=2) 
    
    chain = JUDGE_PROMPT | llm
    
    results = []
    
    for i, item in enumerate(sample_data):
        logger.info(f"Avaliando interação {i+1}/{len(sample_data)}...")
        
        try:
            response = chain.invoke({"query": item["query"], "answer": item["answer"]})
            cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
            
            evaluation = json.loads(cleaned_response)
            results.append(evaluation)
            logger.info(f"Notas atribuídas -> Tom: {evaluation.get('tom_corporativo')} | Concisão: {evaluation.get('concisao')} | Risco: {evaluation.get('alinhamento_risco')}")
            
        except Exception as e:
            logger.error(f"Erro ao avaliar item {i+1}: {e}")
            
        # Proteção estendida contra Rate Limit 
        if i < len(sample_data) - 1:
            time.sleep(6)
            
    if results:
        avg_tom = sum(r.get("tom_corporativo", 0) for r in results) / len(results)
        avg_concisao = sum(r.get("concisao", 0) for r in results) / len(results)
        avg_risco = sum(r.get("alinhamento_risco", 0) for r in results) / len(results)
        
        logger.info(f"\n=== RESULTADOS FINAIS (LLM-as-a-Judge) ===\nTom Corporativo: {avg_tom:.2f}/5.0\nConcisão: {avg_concisao:.2f}/5.0\nPolítica de Risco: {avg_risco:.2f}/5.0\n==========================================")

        logger.info("Enviando métricas de negócio para o MLflow...")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Datathon_LLM_Evaluation")
        
        with mlflow.start_run(run_name="Business_Metrics_Eval"):
            mlflow.log_param("max_samples", max_samples if max_samples else "All_Golden_Set")
            mlflow.log_metrics({
                "biz_tom_corporativo": avg_tom,
                "biz_concisao": avg_concisao,
                "biz_alinhamento_risco": avg_risco
            })

if __name__ == "__main__":
    # Executa a partir da raiz do projeto
    run_llm_judge("data/golden_set/golden_set.json")