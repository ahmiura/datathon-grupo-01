import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Lista de palavras-chave usadas em ataques de Prompt Injection ou fora de escopo
BLOCKED_KEYWORDS = ["ignore", "esqueça tudo", "você é um", "hack", "system prompt", "piada", "receita", "poema"]

def check_input(user_input: str) -> Tuple[bool, str]:
    """Verifica se o input do usuário é seguro e dentro do escopo financeiro."""
    user_input_lower = user_input.lower()
    
    for keyword in BLOCKED_KEYWORDS:
        if keyword in user_input_lower:
            logger.warning(f"Guardrail de Input ativado: tentativa de injeção detectada ({keyword})")
            return False, "Acesso negado: Sua mensagem contém comandos não permitidos ou fora do escopo financeiro."
    
    return True, ""

def check_output(llm_output: str) -> Tuple[bool, str]:
    """Verifica se a saída do LLM vaza dados sensíveis (PII) e os mascara."""
    # Expressão regular básica para detectar formato de CPF (ex: 123.456.789-00)
    cpf_pattern = re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b')
    
    if cpf_pattern.search(llm_output):
        logger.warning("Guardrail de Output ativado: Vazamento de PII (CPF) detectado e mascarado.")
        safe_output = cpf_pattern.sub("***.***.***-**", llm_output)
        return False, safe_output
        
    return True, llm_output