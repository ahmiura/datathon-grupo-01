# Arquitetura da Solução

## Fluxo Online: API, Agente ReAct e RAG

```mermaid
flowchart LR
  User[Usuario] --> API[FastAPI /chat]
  API --> InputGuard[Guardrail de Input]
  InputGuard --> Agent[LangChain AgentExecutor ReAct]

  Agent --> LLM[Google GenAI LLM]
  Agent --> Tools{Tools}

  Tools --> PredictTool[predict_petr4_close_price]
  Tools --> PriceTool[get_current_stock_price]
  Tools --> CalcTool[calculator]
  Tools --> RagTool[search_company_documents]

  PredictTool --> DataLoader[fetch_financial_data]
  PriceTool --> DataLoader
  DataLoader --> FeatureDB[(PostgreSQL features_db stock_prices)]
  DataLoader --> YFinance[yfinance fallback]
  PredictTool --> LSTM[LSTM Predictor PyTorch]

  CalcTool --> Math[Validacao de expressao numerica]

  RagTool --> Retriever[query_documents k=3]
  Retriever --> FAISS[(FAISS Vector Store)]
  FAISS --> Embeddings[Google Generative AI Embeddings]
  PDFs[PDFs em data/raw] --> Loader[PyPDFDirectoryLoader]
  Loader --> Splitter[RecursiveCharacterTextSplitter chunk 1000 overlap 200]
  Splitter --> Embeddings
  Embeddings --> FAISS

  Agent --> OutputGuard[Guardrail de Output PII]
  OutputGuard --> Response[Resposta final]
  Response --> User

  API --> Prometheus[Prometheus /metrics]
  Agent --> Langfuse[Langfuse traces]
```

## Fluxo MLOps: Treino, Avaliação e Drift

```mermaid
flowchart TD
  Airflow[Apache Airflow] --> FeatureTask[Sincronizar Feature Store PostgreSQL]
  FeatureTask --> FeatureDB[(PostgreSQL features_db stock_prices)]
  FeatureTask --> Parquet[(Snapshot Parquet derivado)]
  FeatureDB --> Train[Treinar LSTM Grid Search]
  Parquet --> Train
  Train --> MLflow[(MLflow Tracking)]
  Train --> ModelFiles[models/lstm_model.pth scaler.joblib hyperparameters.json]

  Airflow --> RagUpdate[Atualizar RAG Context]
  RagUpdate --> PDFs[PDFs data/raw]
  PDFs --> Splitter[Chunking 1000 overlap 200]
  Splitter --> Embed[Embeddings Gemini]
  Embed --> VectorIndex[(FAISS Index)]

  Airflow --> RagasEval[RAGAS Technical Eval]
  GoldenSet[(Golden Set JSON)] --> RagasEval
  RagasEval --> RagasMetrics[Faithfulness Answer Relevancy Context Precision Context Recall]
  RagasMetrics --> MLflow

  Airflow --> BizJudge[LLM as a Judge]
  GoldenSet --> BizJudge
  BizJudge --> BizMetrics[Tom Corporativo Concisao Alinhamento Risco]
  BizMetrics --> MLflow

  Airflow --> Drift[Evidently Drift Detection]
  Drift --> FeatureDB
  Drift --> DriftReport[reports/data_drift_report.html]
  Drift --> DriftMetric[dataset_drift_detected]
  DriftReport --> MLflow
  DriftMetric --> MLflow
  DriftMetric -->|drift detectado| Train

  MLflow --> MLflowDB[(PostgreSQL mlflow_db)]
  Airflow --> AirflowDB[(PostgreSQL airflow_db)]
```

## Separação de Responsabilidades

| Camada | Componente | Responsabilidade |
| --- | --- | --- |
| Serving | FastAPI | Expor `/chat`, `/predict` e `/metrics` |
| Orquestração | LangChain ReAct | Selecionar tools e consolidar a resposta |
| Dados financeiros | `fetch_financial_data` | Sincronizar PostgreSQL canônico e fallback para `yfinance` |
| Feature/cache store | `postgres_features/features_db` | Fonte de verdade para API, agente, drift e treino |
| Snapshot de treino | `data/processed/feature_store.parquet` | Artefato derivado para auditoria, DVC e reprodutibilidade |
| RAG | FAISS + Gemini Embeddings | Busca semântica em relatórios PDF |
| Modelo preditivo | PyTorch LSTM | Prever fechamento de PETR4.SA |
| Avaliação técnica | RAGAS | Medir fidelidade, relevância e recuperação de contexto |
| Avaliação de negócio | LLM-as-a-Judge | Medir tom, concisão e alinhamento à política de risco |
| Tracking | MLflow | Registrar métricas, parâmetros e artefatos |
| Drift | Evidently | Detectar mudanças em `Close`, `Volume`, `Returns`, `Volatility` |
| Observabilidade | Prometheus, Grafana, Langfuse | Métricas de API e traces de LLM |

## Detalhes de Implementação

- **Vector Store (FAISS):** implementado em `data/processed/faiss_index` e carregado por `src/agent/rag_pipeline.py`.
- **Modelo de Embeddings:** variável de ambiente `GEMINI_EMBEDDING_MODEL` (padrão `models/gemini-embedding-001`).
- **LLM Orquestrador:** variável de ambiente `GEMINI_MODEL_NAME` (ex.: `gemma-3-27b-it`), usado pelo agente via Google GenAI.
- **MLflow Tracking:** controlado por `MLFLOW_TRACKING_URI` (no compose aponta para `http://mlflow:5000`); sem variável, o treino usa fallback local `mlruns/`.
- **Docker Compose (serviços-chave):** `api`, `airflow`, `mlflow`, `postgres_features`, `postgres_db`, `postgres_airflow`, `prometheus`, `grafana`, `loki`.
- **Health/Readiness:** API expõe `/health`, `/ready` e `/metrics` (Prometheus). O `docker-compose.yml` usa `/ready` como healthcheck.
- **Observabilidade LLM:** `langfuse` integrado via callback em `src/serving/app.py` para traces do agente.
- **Drift & Avaliação:** scripts de drift (`monitoring/drift.py`) e avaliação (`evaluation/ragas_eval.py`, `evaluation/llm_judge.py`) são executados pelas DAGs do Airflow.

