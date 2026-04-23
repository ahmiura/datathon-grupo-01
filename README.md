# Tech Challenge Fase 5 (Datathon) - Financial AI Agent & Stock Predictor

Projeto integrador desenvolvido para o Datathon da Fase 5. Esta aplicação evoluiu de um modelo preditivo clássico para um **Agente de Inteligência Artificial Baseado em LLMs (ReAct)**, integrando Previsão de Séries Temporais (LSTM), Recuperação de Contexto (RAG), MLOps Nível 2 e Guardrails de Segurança.

## 🌟 Destaques do Projeto
- **Agente Autônomo (ReAct):** LLM orquestrador (Google Gemma 3) capaz de raciocinar e utilizar 4 ferramentas exclusivas (Calculadora, Busca de Preço em Tempo Real, Previsão LSTM e Busca em Documentos).
- **Pipeline RAG Local:** Ingestão de relatórios financeiros em PDF e vetorização usando FAISS e embeddings do Google.
- **Segurança e Governança:** Mapeamento OWASP implementado via Guardrails (Bloqueio de Prompt Injection e mascaramento de PII/CPF).
- **Avaliação Contínua:** Pipeline automatizado (via Airflow) avaliando a qualidade técnica via **RAGAS** e métricas de negócio (Tom, Concisão e Risco) via um **LLM-as-a-Judge customizado**.
- **Engenharia de Software:** API FastAPI 100% tipada, dependências gerenciadas via `pyproject.toml` e **cobertura de testes unitários >80%** com `pytest`.
- **MLOps Nível 2:** Rastreamento de experimentos (MLflow), Observabilidade de LLMs (Langfuse), Monitoramento de Infra (Prometheus/Grafana) e tracking de artefatos de governança (Model Cards e System Cards).
- **Detecção de Drift Automática (Evidently):** Monitoramento contínuo de estabilidade populacional (PSI) e *Data Drift* entre distribuições de treino e inferência, gerando alertas preditivos de degradação.
- **Orquestração (Continuous Training com Airflow):** DAG customizada (`datathon_mlops_continuous_training`) executando o seguinte pipeline end-to-end:
  1. **Atualização da Feature Store:** Ingestão incremental de novos dados financeiros em store operacional separada do backend do MLflow.
  2. **Retreinamento LSTM:** Busca de hiperparâmetros (Grid Search), treino e seleção do melhor modelo via MLflow.
  3. **Atualização RAG:** Ingestão e vetorização de novos documentos corporativos no FAISS.
  4. **Avaliação Técnica:** LLM-as-a-Judge rodando framework **RAGAS** (Fidelidade, Relevância, Precisão).
  5. **Avaliação de Negócios:** LLM-as-a-Judge validando métricas de Tom, Concisão e Risco nas respostas.
- **Governança de Dados (DVC & Synthetic Data):** Utilização do DVC para rastreabilidade de artefatos de dados e geração de dados sintéticos via `Faker` para proteção de PII no ambiente de desenvolvimento local.
- **Análise Exploratória (EDA):** Notebook focado em análise temporal de ativos, justificando arquiteturas de modelagem com base em regimes de volatilidade.

## 📋 Estrutura
- `src/agent/`: Coração cognitivo (Prompt ReAct, Tools customizadas e Pipeline RAG).
- `src/models/`: Arquitetura do PyTorch, script de treino e integração com MLflow.
- `src/serving/`: Aplicação FastAPI com injeção de Middlewares e Guardrails.
- `dags/`: Pipelines de orquestração para o Apache Airflow.
- `security/`: Implementações de mitigação OWASP (Input/Output).
- `monitoring/`: Scripts de observabilidade de dados (Detecção de Drift com Evidently e logs estruturados).
- `evaluation/`: Scripts de avaliação RAGAS (Faithfulness, Relevancy, etc).
- `tests/`: Suíte de testes automatizados (`pytest`).
- `docs/`: Artefatos de governança (OWASP Mapping, System Card).
- `docs/ARCHITECTURE.md`: Diagramas Mermaid do fluxo online e dos pipelines MLOps.

## Arquitetura de Serviços

O `docker-compose.yml` separa os bancos por responsabilidade:

- `postgres_db`: backend store do MLflow (`mlflow_db`), usado apenas para experimentos, runs, métricas e parâmetros.
- `postgres_features`: feature/cache store operacional (`features_db`), onde o `data_loader.py` materializa a tabela `stock_prices`.
- `postgres_airflow`: metadados internos do Apache Airflow (`airflow_db`).
- `mlflow`: UI e tracking server em `http://localhost:5000`.
- `api`: API FastAPI em `http://localhost:8000`.
- `airflow`: DAGs de treinamento, avaliação e drift em `http://localhost:8080`.
- `prometheus`, `grafana` e `loki`: observabilidade de infraestrutura e aplicação.

## 🚀 Como Rodar

### Pré-requisitos
- Docker e Docker Compose instalados.
- Chaves de API do Google Gemini e do Langfuse configuradas em um arquivo `.env`.
- Use `.env.example` como modelo. Não versionar o `.env` real.

### 1. Execução Completa (Docker)
O ambiente está containerizado. Para subir a API, o MLflow, Prometheus e Grafana:

```bash
docker compose up --build -d
```

### 2. Acessando os Serviços
- **API Docs (Swagger):** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Apache Airflow:** http://localhost:8080 (Login: `admin` / Senha: `admin`)
- **Grafana:** http://localhost:3000 (Login: `admin` / Senha: `admin`)
- **Langfuse:** https://cloud.langfuse.com/ Acessado via Cloud para telemetria e observabilidade do Agente LLM.
- **Prometheus:** http://localhost:9090

---

## 🧪 Como Testar a API

### Opção A: Chat com o Agente de IA (Etapa 2 e 4)
Converse naturalmente com a IA financeira. Ela decidirá automaticamente qual ferramenta acionar (Previsão, Cotação, PDF ou Calculadora) e passará pelos Guardrails de Segurança.

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Qual é a previsão da PETR4 para amanhã e o que diz o relatório sobre os dividendos?"}'
```

### Opção B: Previsão Manual
Envie uma lista com os preços de fechamento dos últimos 60 dias.

```json
{
  "last_60_days": [34.5, 35.2, 34.8, ..., 36.1]
}
```

### Opção C: Teste de Carga (Gerar Métricas)
Para ver os gráficos do Grafana se moverem, execute o script de teste de carga em outro terminal (requer python local):

```bash
pip install requests
python load_test.py
```
*Isso enviará requisições aleatórias para a API, simulando uso real.*

---

## 🧠 Treinamento do Modelo
Caso queira retreinar o modelo de Deep Learning (LSTM) do zero usando o ambiente configurado no Docker:

```bash
docker compose exec api python src/models/train.py
```
Isso executará o **Grid Search**, salvará o melhor modelo em `models/` e registrará os resultados no MLflow.

## Pipelines MLOps

### Continuous Training

A DAG `datathon_mlops_continuous_training` executa:

1. Atualização incremental da feature store local em `data/processed/feature_store.parquet`.
2. Treinamento LSTM com registro no MLflow.
3. Atualização do índice FAISS para RAG.
4. Avaliação técnica RAGAS.
5. Avaliação de negócio com LLM-as-a-Judge.

### Drift Detection e Retreinamento

A DAG `drift_detection_and_retraining` executa `monitoring/drift.py`, compara dados de referência com dados recentes usando Evidently e salva:

- Relatório HTML em `reports/data_drift_report.html`.
- Métrica `dataset_drift_detected` e artefato no MLflow.
- Retreinamento automático quando drift é detectado.

O script usa `postgres_features` como cache de cotações (`stock_prices`). Se esse banco estiver temporariamente indisponível, o `data_loader.py` tenta fallback direto para `yfinance`; nesse caso o pipeline pode continuar, mas sem aproveitar o cache.

## Variáveis de Ambiente

Principais variáveis usadas pelos containers:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000

FEATURE_DB_HOST=postgres_features
FEATURE_DB_PORT=5432
FEATURE_DB_NAME=features_db
FEATURE_DB_USER=features
FEATURE_DB_PASS=features

GOOGLE_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
GEMINI_MODEL_NAME=gemma-3-27b-it
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

As variáveis legadas `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` e `DB_PASS` ainda são aceitas pelo `data_loader.py` como fallback para execuções locais antigas, mas o compose usa `FEATURE_DB_*`.

## Validação Local

Com o ambiente em execução:

```bash
docker compose ps
docker compose exec api pytest tests/
docker compose exec api python src/models/train.py
docker compose exec airflow python monitoring/drift.py
```

Para validar a configuração sem subir serviços:

```bash
docker compose config
python -m compileall src monitoring evaluation dags
```
