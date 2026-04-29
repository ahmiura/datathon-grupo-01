# System Card: Financial AI Agent & Predictor

## 1. Detalhes do Sistema
* **Nome do Sistema:** Datathon - Agente Financeiro Inteligente
* **Versão:** 1.0.0
* **Data da Última Atualização:** 21/04/2026
* **Desenvolvedores:** Alex Hideki Miura
* **Licença:** Uso Acadêmico / Interno

## 2. Uso Pretendido (Intended Use)
* **Objetivo Primário:** Auxiliar usuários a obterem informações atualizadas e previsões sobre ativos financeiros (com foco na PETR4) e extrair insights de relatórios internos de forma conversacional e intuitiva.
* **Casos de Uso Aprovados:** 
  * Busca de cotações em tempo real via Yahoo Finance.
  * Inferência de preços de fechamento para o próximo dia útil usando rede neural (LSTM).
  * Questionamentos contextuais baseados em documentos internos em formato PDF (RAG).
  * Cálculos matemáticos estruturados.
* **Casos de Uso Não Aprovados:** Aconselhamento financeiro real para investimentos de risco, geração de conteúdo não-financeiro, execução de código arbitrário e processamento de informações pessoais não anonimizadas.

## 3. Arquitetura do Sistema
O sistema utiliza uma arquitetura orquestrada (ReAct) composta pelos seguintes módulos:
* **LLM Core:** `gemma-3-27b-it` (família Open-Weights do Google via GenAI API) servindo como cérebro orquestrador.
* **Retrieval-Augmented Generation (RAG):** Documentos particionados (chunking) via `LangChain`, vetorizados usando `gemini-embedding-001` e armazenados localmente via `FAISS`.
* **Modelo Preditivo Subjacente:** Rede Neural Recorrente (LSTM) treinada em `PyTorch`, monitorada por `MLflow`.
* **Feature/Cache Store:** PostgreSQL dedicado (`postgres_features/features_db`) para materializar cotações em `stock_prices`, separado do banco de metadados do MLflow.
* **Serving:** API em `FastAPI`, provisionada em container `Docker`, com endpoint de hot-reload (`/reload-model`) para atualização de pesos sem downtime.
* **Tracking e Observabilidade:** MLflow usa um PostgreSQL próprio (`postgres_db/mlflow_db`) para metadados de experimentos. Métricas e traces adicionais são capturados via `Langfuse`, `Prometheus` e `Grafana`.

## 4. Limitações e Fatores de Risco
* **Dependências Externas e Rate Limits:** O sistema consome APIs de terceiros (Yahoo Finance e Google GenAI). Embora a arquitetura possua mecanismos de resiliência e *Circuit Breakers* (como o cache na Feature Store em PostgreSQL e fallbacks para dados sintéticos), indisponibilidades prolongadas dessas APIs podem atrasar a inferência ou limitar a atualização das cotações em tempo real.
* **Alucinações (Hallucinations):** Apesar do framework RAG ater o modelo aos fatos recuperados dos documentos (Grounding) e do uso de ferramentas rígidas, o LLM ainda pode extrapolar respostas ou errar lógicas financeiras caso falhe em acionar a calculadora. Esse risco é minimizado e monitorado continuamente pelo pipeline de avaliação automatizado (RAGAS).
* **Viés Temporal (Data Drift):** O modelo LSTM integrado é sensível a quebras estruturais e volatilidade atípica do mercado. Para mitigar o impacto de mudanças macroeconômicas na acurácia (RMSE/MAE), o sistema conta com uma DAG de *Drift Detection* (via Evidently) que monitora a distribuição dos dados e aciona o retreinamento (Continuous Training) automaticamente quando necessário.

## 5. Medidas de Segurança, Privacidade e Guardrails (OWASP)
Para mitigar riscos comuns em aplicações com LLMs, este sistema implementa:
* **Guardrails de Input:** Bloqueio ativo de tentativas de `Prompt Injection` e restrição de escopo temático (rejeição de pedidos de piadas, receitas, comandos de sistema).
* **Guardrails de Output:** Expressões Regulares (Regex) varrem o conteúdo devolvido pelo Agente mascarando e omitindo Dados Pessoalmente Identificáveis (PII) e informações sensíveis (como CPFs, CNPJs, E-mails e Contas Bancárias) para proteção da LGPD e sigilo bancário.
* **Validação de Tools:** As ferramentas de execução matemática rejeitam comandos com caracteres alfabéticos para mitigar Injeção de Código Arbitrário.

## 6. Avaliação e Performance
A qualidade das respostas do Agente é auditada periodicamente através do framework **RAGAS** (LLM-as-a-judge), que gera métricas padronizadas utilizando um conjunto "Golden Set" de validação:

* **Faithfulness (Fidelidade):** Mede se a resposta gerada é fiel aos fatos extraídos dos PDFs.
* **Answer Relevancy (Relevância):** Mede quão direta e pertinente a resposta foi à pergunta do usuário.
* **Context Precision (Precisão do Contexto):** Avalia se a ferramenta de busca (FAISS) priorizou as passagens corretas.
* **Context Recall (Abrangência):** Mede se nenhuma informação crucial para a resposta foi esquecida no processo de busca.

> *Nota:* A execução do pipeline de testes garante cobertura superior a 60% do código fonte base da aplicação.
