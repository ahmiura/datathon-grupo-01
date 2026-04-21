# Mapeamento de Segurança (OWASP Top 10 for LLMs)

Este documento detalha como a API do Agente ReAct (Fase 5) mitiga **5 das principais vulnerabilidades** listadas no OWASP Top 10 para Aplicações baseadas em LLMs. A seleção destas 5 ameaças foca nos vetores de ataque mais críticos e aplicáveis ao contexto financeiro da nossa arquitetura (cumprindo os requisitos exigidos do Datathon), sendo muitas delas mitigadas através do nosso módulo `security/guardrails.py`.

## 1. LLM01: Prompt Injection
* **Ameaça:** Usuários mal-intencionados podem tentar sobrescrever as instruções do sistema (ex: "Ignore as regras anteriores e faça X").
* **Mitigação Implementada:** Nossa API possui um **Guardrail de Input** (`check_input`) que intercepta a requisição antes de enviá-la ao LLM, bloqueando palavras-chave bloqueadas (como "ignore", "system prompt") e retornando Erro 403.

## 2. LLM02: Insecure Output Handling
* **Ameaça:** A saída do LLM pode vazar dados confidenciais ou códigos maliciosos que serão executados pelo frontend.
* **Mitigação Implementada:** Nossa API conta com um **Guardrail de Output** (`check_output`). Todas as respostas geradas pelo Agente são analisadas por RegEx antes de chegar ao usuário, mascarando PIIs/CPFs.

## 3. LLM06: Sensitive Information Disclosure
* **Ameaça:** O Agente pode memorizar e vazar dados pessoais (PII), como CPFs contidos nos documentos ou injetados acidentalmente.
* **Mitigação Implementada:** O Guardrail de saída detecta ativamente o padrão matemático de CPFs (`\b\d{3}\.\d{3}\.\d{3}-\d{2}\b`) e mascara os números com asteriscos (`***.***.***-**`) antes de expor a resposta.

## 4. LLM07: Insecure Plugin Design
* **Ameaça:** Ferramentas (Tools) do agente sem validação adequada, permitindo execução de código arbitrário.
* **Mitigação Implementada:** A nossa Tool `calculator` possui validação estrita. Ela analisa a expressão matemática e bloqueia qualquer string que contenha letras ou comandos não numéricos (como `import os`), impedindo injeção de código Python.

## 5. LLM09: Overreliance (Alucinações)
* **Ameaça:** Confiar cegamente nas respostas do LLM, que pode inventar informações financeiras ou preços irreais.
* **Mitigação Implementada:** O sistema utiliza arquitetura RAG e ferramentas rígidas. Para preços atuais, a ferramenta extrai a cotação real da B3 via Yahoo Finance. Para preços futuros, consulta um modelo determinístico PyTorch (LSTM). Além disso, temos um pipeline diário de avaliação com RAGAS para monitorar a Fidelidade (Faithfulness).