# Model Card: PETR4_LSTM_Predictor

Este Model Card fornece os detalhes técnicos, dados de treinamento, métricas de avaliação e considerações éticas do modelo preditivo subjacente utilizado pela nossa ferramenta de Agente de IA.

## 1. Detalhes do Modelo
* **Nome do Modelo:** `PETR4_LSTM_Predictor`
* **Versão:** `v1.0.0`
* **Tipo de Modelo:** Rede Neural Recorrente (RNN) com arquitetura Long Short-Term Memory (LSTM).
* **Framework:** PyTorch
* **Desenvolvedores:** Alex Hideki Miura
* **Licença:** Uso Acadêmico / Interno

## 2. Uso Pretendido (Intended Use)
* **Uso Primário:** Prever o preço de fechamento (Close) da ação da Petrobras (`PETR4.SA`) para o próximo dia útil (t+1) com base em uma janela histórica de preços. O modelo é executado através do Agente ReAct utilizando a tool `predict_petr4_close_price`.
* **Casos de Uso Aprovados:** 
  * Simulações e estudos de tendências financeiras.
  * Respostas conversacionais com ressalva de que se trata de uma previsão orientada por IA.
* **Casos de Uso Fora de Escopo:** 
  * Operações de *High-Frequency Trading (HFT)* automatizadas.
  * Aconselhamento de investimentos garantidos sem supervisão humana.

## 3. Dados de Treinamento e Avaliação
* **Fonte de Dados:** Yahoo Finance API (`yfinance`), materializada em uma Feature/Cache Store dedicada em PostgreSQL (`postgres_features/features_db`) para evitar gargalos, reduzir chamadas externas e manter os dados operacionais separados do backend store do MLflow.
* **Ativo (Ticker):** `PETR4.SA` (Petrobras PN)
* **Feature Utilizada:** Preço de Fechamento Histórico (`Close`).
* **Pré-processamento:**
  * Imputação/Limpeza de valores nulos ou anômalos.
  * Escalonamento dos dados (ex: `MinMaxScaler`) para o intervalo `[0, 1]` visando convergência eficiente dos gradientes.
  * Janelamento de tempo (Time-steps): Sequências estáticas de **60 dias** (`TIMESTEPS = 60`) predizendo 1 dia no futuro.

## 4. Arquitetura do Modelo e Hiperparâmetros
O modelo foi selecionado através de uma técnica de **Grid Search** automatizada que explora múltiplas topologias. A configuração exata do melhor modelo (salva dinamicamente em `model_hyperparameters.json`) é extraída do seguinte espaço de busca:

* **Hidden Size:** 50, 64, 100, 128
* **Num Layers (Camadas LSTM):** 1 a 2
* **Dropout:** 0.1 a 0.4
* **Função de Perda:** Mean Squared Error (MSELoss)
* **Otimizador:** Adam (Learning Rate entre 0.0005 e 0.005)
* **Épocas:** 40 a 80

> *O modelo recebe tensores no formato `(batch_size, 60, 1)` e retorna `(batch_size, 1)`.*

## 5. Governança e Rastreamento (MLflow)
Todos os ciclos de treinamento são versionados e registrados no **MLflow**, garantindo o **Nível 2 de MLOps**. As seguintes tags obrigatórias são registradas para conformidade corporativa:

* `model_name`: PETR4_LSTM_Predictor
* `model_version`: v1.0.0
* `training_data_version`: v1_initial
* `model_type`: LSTM
* `risk_level`: medium
* `owner`: grupo-01
* `phase`: datathon-fase05
* `fairness_checked`: true

## 6. Métricas de Avaliação (Quantitative Analyses)
O modelo é avaliado contra um subconjunto de testes (Test Split) utilizando métricas clássicas de regressão. O tracking do MLflow registra:
* **RMSE (Root Mean Squared Error):** Penaliza erros mais grosseiros na previsão.
* **MAE (Mean Absolute Error):** Média absoluta do desvio de preço em Reais (R$).
* **MAPE (Mean Absolute Percentage Error):** Erro percentual da previsão em relação ao preço real do ativo.
* **Gráfico de Predição:** Uma plotagem (`prediction_plot.png`) comparando a curva de fechamento real e a curva predita pela IA é salva como artefato no MLflow.

## 7. Fatores de Risco e Limitações
* **Volatilidade:** Modelos LSTM treinados apenas com preços históricos de fechamento não capturam eventos imprevisíveis como pandemias, trocas de comando da estatal ou crises geopolíticas.
* **Data Drift:** Se houver uma quebra estrutural na tendência de mercado da Petrobras, a performance do modelo (RMSE) degradará. Este risco é mitigado pela orquestração do **Apache Airflow**, que prevê o retreinamento contínuo (Continuous Training) do modelo na ocorrência de *drifts* detectados pelo Evidently.
* **Circuit Breaker / Fallback:** Caso a consulta ao PostgreSQL operacional falhe, o `data_loader.py` tenta fallback direto para `yfinance`. Caso a extração externa também falhe por rate limit ou conectividade, o pipeline evita corromper a feature store e retorna dados vazios para que a etapa chamadora aborte ou acione o fallback de aplicação.
