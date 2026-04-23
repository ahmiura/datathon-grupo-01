# Relatório de Revisão de Código — Datathon Grupo 01

Data: 23/04/2026

Este documento consolida a revisão do código-chave do repositório, descreve mudanças realizadas, sugestões de PRs e um checklist acionável para entrega.

**Escopo da Revisão**
- Componentes analisados: `src/` (models, serving, agent, data, features), `security/`, `tests/`, `dags/`, `monitoring/` e `docs/`.
- Execução de testes dentro do container `api` para validar correções.

**Resumo Executivo**
- Corrigi uma falha de coleta de testes causada por um script de carga em loop (`load_test.py`). Renomeado para `load_test_script.py` para não ser coletado pelo `pytest`.
- Substituí uso inseguro de `eval()` na ferramenta `calculator` por um avaliador AST com `Decimal`, adicionando suporte a `%` e `//` e preservando formato decimal quando esperado.
- Instanciei o agente ReAct uma vez no `lifespan` do FastAPI e reutilizei no endpoint `/chat` para reduzir latência.
- Adicionei endpoints de observabilidade básicos: `/health` e `/ready`.
- Criei/atualizei configurações de teste: `pytest.ini`, seção em `pyproject.toml`, e instruções no `README.md` para execução em container e marcação `integration`.

**Arquivos modificados / adicionados**
- Ferramenta e segurança:
  - [src/agent/tools.py](src/agent/tools.py) — refatoração do `calculator` (AST + Decimal + operadores adicionais).
  - [tests/test_tools.py](tests/test_tools.py) — novos testes para `calculator`.
- Servidor / Orquestração:
  - [src/serving/app.py](src/serving/app.py) — instanciamento do agente no `lifespan`, uso do agente reutilizável no `/chat`, adição de `/health` e `/ready`.
- Testes / Infra docs:
  - [pytest.ini](pytest.ini) — marcador `integration`.
  - [pyproject.toml](pyproject.toml) — opções pytest padronizadas.
  - [README.md](README.md) — instruções para rodar testes em container/CI e nota sobre `load_test_script.py`.
  - `load_test.py` removido e [load_test_script.py](load_test_script.py) adicionado (não coletável).

**Testes executados**
- Testes isolados por arquivo rodaram com sucesso dentro do container `api`.
- Após mudanças, executei `tests/test_tools.py`, `tests/test_api.py`, `tests/test_agent.py` e a bateria completa por arquivo — todos os testes passaram.

**PRs sugeridos (pequenos e temáticos)**
- PR 1 — "fix(calculator): secure AST evaluator + Decimal support"
  - Alterações: [src/agent/tools.py](src/agent/tools.py), [tests/test_tools.py](tests/test_tools.py)
  - Objetivo: remover risco de RCE, melhor precisão, cobrir com testes.

- PR 2 — "feat(serving): instantiate agent in lifespan + readiness endpoints"
  - Alterações: [src/serving/app.py](src/serving/app.py)
  - Objetivo: reduzir latência do `/chat`, melhorar readiness e health checks.

- PR 3 — "chore(tests): pytest config and README test docs"
  - Alterações: [pytest.ini](pytest.ini), [pyproject.toml](pyproject.toml), [README.md](README.md)
  - Objetivo: padronizar execução de testes e instruir uso em container/CI.

- PR 4 — "fix(testing): exclude long-running load_test from pytest collection"
  - Alterações: remoção de `load_test.py`, adição de `load_test_script.py` no repositório (documentado no README).

Cada PR deve ser pequeno, com um único objetivo, conter descrição clara e referência ao issue (se aplicável).

**Checklist obrigatório antes da entrega / merge**
1. Executar suíte completa de testes em runner com recursos adequados (recomendado: container com >=4GB RAM). Comandos:

```bash
docker compose up -d
docker compose exec api python -m pytest -q
```

2. Adicionar workflow CI (GitHub Actions / similar) que:
   - sobe serviços essenciais com `docker compose up --build -d`;
   - executa `python -m pytest -q` (separar unit vs integration via markers);
   - publica cobertura (pytest-cov) e artifacts relevantes.

3. Garantir rastreabilidade de artefatos de treino:
   - salvar `scaler.joblib` e `model_hyperparameters.json` como artifacts no MLflow (ou DVC);
   - versionar snapshots de dados (`data/processed/feature_store.parquet`) com DVC se aplicável.

4. Melhorias operacionais recomendadas (prioridade média):
   - `src/data/data_loader.py`: tornar `COOLDOWN_SECONDS` e timeouts configuráveis via env; implementar backoff exponencial e retries com limites.
   - `src/features/data.py`: adicionar validações para casos com poucos dados e mensagens de erro amigáveis.
   - `security/guardrails.py`: ampliar detecção de PII (e-mail, CNPJ, contas bancárias) e adicionar testes unitários.
   - Logging: padronizar logs estruturados (JSON), incluir `request_id` e `trace_id` para rastreamento distribuído.

5. Segurança e auditoria:
   - Executar scan estático (bandit, semgrep) para garantir ausência de padrões perigosos.
   - Testar guardrails com casos adversariais de prompt injection e PII.

**Checklist opcional (valor agregado)**
- Implementar endpoint de `metrics/health` que agregue checks do PostgreSQL, FAISS index e MLflow.
- Adicionar teste de carga controlado (com tempo limitado) em CI que gere métricas e não bloqueie a suíte.
- Documentar política de secrets e instruções para `.env` e `.env.example` no README.

**Observações finais e riscos**
- Dependências de modelos/embeddings externos (Google GenAI / Langfuse) podem falhar em CI; sempre permita modos degradados e mocks para testes automatizados.
- Testes desenvolvidos passaram isoladamente; rodar a suíte inteira no CI com recursos adequados é recomendado para validar integração completa.

Se desejar, eu posso:
- Gerar os PRs sugeridos localmente e abrir branches com commits prontos; ou
- Implementar as melhorias prioritárias apontadas (ex.: backoff no `data_loader` e testes para `guardrails`).

---
Arquivos-chave alterados nesta revisão: [src/agent/tools.py](src/agent/tools.py), [src/serving/app.py](src/serving/app.py), [tests/test_tools.py](tests/test_tools.py), [pytest.ini](pytest.ini), [pyproject.toml](pyproject.toml), [README.md](README.md), [load_test_script.py](load_test_script.py).

*** Fim do relatório ***
