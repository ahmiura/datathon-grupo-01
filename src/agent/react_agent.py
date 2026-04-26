import logging
import os
from typing import Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.tools import predict_petr4_close_price, get_current_stock_price, calculator, search_company_documents

logger = logging.getLogger(__name__)

# Prompt seguindo o framework ReAct do LangChain
REACT_PROMPT = PromptTemplate.from_template("""Responda às perguntas do usuário da melhor forma possível, agindo como um consultor financeiro. 
Você tem acesso às seguintes ferramentas:

{tools}

Use EXATAMENTE o seguinte formato:

Question: a pergunta que você deve responder
Thought: você deve sempre pensar passo a passo sobre o que fazer
Action: a ação a ser tomada, deve ser UMA de [{tool_names}]
Action Input: a entrada necessária para a ação
Observation: o resultado da ação executada
... (os passos Thought/Action/Action Input/Observation podem se repetir N vezes)
Thought: Eu agora sei a resposta final
Final Answer: a resposta final e clara para a pergunta do usuário

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

def create_datathon_agent(model_name: Optional[str] = None, temperature: float = 0.0) -> AgentExecutor:
    """
    Cria e inicializa o Agente ReAct (Raciocínio e Ação) financeiro.

    Args:
        model_name (Optional[str]): Nome do modelo LLM (ex: gemini-2.0-flash). Se None, usa variável de ambiente.
        temperature (float): Grau de aleatoriedade das respostas. Padrão é 0.0 para maior precisão financeira.

    Returns:
        AgentExecutor: Executor do LangChain configurado com prompt, ferramentas e políticas de retry.
    """
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemma-3-27b-it")
        
    tools = [predict_petr4_close_price, get_current_stock_price, calculator, search_company_documents]
    
    if len(tools) < 3:
        logger.warning("Datathon exige >= 3 tools. Fornecidas: %d", len(tools))
        
    # max_retries=3 permite que o LangChain espere e tente novamente em caso de instabilidade (Erro 503 - High Demand)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_retries=3)
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    
    return AgentExecutor(agent=agent, tools=tools, max_iterations=15, verbose=True, handle_parsing_errors=True)