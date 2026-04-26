# Plano de Conformidade com a LGPD

Este documento descreve as medidas adotadas no sistema do Agente Financeiro (Fase 05) para garantir a privacidade dos dados e a conformidade com a Lei Geral de Proteção de Dados (LGPD).

## 1. Minimização de Coleta de Dados
A API de previsão e conversação (`/chat` e `/predict`) opera em modo *stateless* (sem estado). Não exigimos autenticação de usuário, não coletamos logs de IP e não armazenamos o histórico de mensagens em bancos de dados relacionais atrelados a identidades de pessoas físicas.

## 2. Mascaramento Dinâmico de PII (Personally Identifiable Information)
Devido à natureza não-determinística dos LLMs, existe o risco de o Agente vazar ou repassar dados sensíveis presentes acidentalmente em relatórios corporativos ou no próprio input do usuário.
* **Implementação:** Desenvolvemos um **Guardrail de Output** (`security/guardrails.py`) que processa a string final antes de enviá-la via rede.
* **Mecanismo:** Utiliza expressões regulares (Regex) para identificar padrões de dados pessoais, corporativos e financeiros (como CPFs, CNPJs, E-mails e Contas Bancárias) e aplica máscaras de ofuscação (ex: `***.***.***-**` para CPF e `[EMAIL:***@***]`), garantindo que o dado não trafegue para o frontend.

## 3. Rastreabilidade Anonimizada
Para fins de observabilidade operacional e cálculo de métricas RAGAS, o sistema retém logs no Langfuse e MLflow.
* Nenhuma variável que permita a identificação direta do titular é enviada nas tags ou metadados de telemetria.
* O processamento e vetorização de documentos internos (RAG) no FAISS é estritamente limitado a relatórios públicos financeiros (B3) ou corporativos não-pessoais.

## 4. Direito ao Esquecimento
Por ser uma arquitetura que não retém memória persistente do usuário entre sessões (sem banco de sessões), o direito ao apagamento é garantido nativamente pelo ciclo de vida efêmero dos containers Docker da aplicação.