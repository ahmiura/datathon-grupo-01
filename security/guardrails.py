import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Lista de palavras-chave usadas em ataques de Prompt Injection ou fora de escopo
BLOCKED_KEYWORDS = ["ignore", "esqueça tudo", "você é um", "hack", "system prompt", "piada", "receita", "poema"]

# Padrões para detecção de PII comuns no contexto brasileiro
CPF_PATTERN = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
CNPJ_PATTERN = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
# Padrão simples para detectar números de conta/agência ou sequências longas de dígitos
BANK_ACCOUNT_PATTERN = re.compile(r"\b(?:ag[eê]ncia[:\s]*\d{1,6}[\s,-]*conta[:\s]*\d{1,10}|\d{6,12}-?\d{0,2})\b", re.IGNORECASE)

def check_input(user_input: str) -> Tuple[bool, str]:
    """Verifica se o input do usuário é seguro e dentro do escopo financeiro."""
    user_input_lower = user_input.lower()

    for keyword in BLOCKED_KEYWORDS:
        if keyword in user_input_lower:
            logger.warning(f"Guardrail de Input ativado: tentativa de injeção detectada ({keyword})")
            return False, "Acesso negado: Sua mensagem contém comandos não permitidos ou fora do escopo financeiro."

    return True, ""


def _mask_pii(text: str) -> Tuple[bool, str]:
    """Detecta e mascara PII em texto. Retorna (encontrado, texto_mascarado)."""
    found = False

    # Emails -> [EMAIL:***@***]
    def _email_repl(m):
        nonlocal found
        found = True
        return "[EMAIL:***@***]"

    text = EMAIL_PATTERN.sub(_email_repl, text)

    # CPF -> ***.***.***-** (mantém formato)
    def _cpf_repl(m):
        nonlocal found
        found = True
        return "***.***.***-**"

    text = CPF_PATTERN.sub(_cpf_repl, text)

    # CNPJ -> **.***.***/****-** (mantém formato)
    def _cnpj_repl(m):
        nonlocal found
        found = True
        return "**.***.***/****-**"

    text = CNPJ_PATTERN.sub(_cnpj_repl, text)

    # Contas bancárias / sequências longas de dígitos
    def _bank_repl(m):
        nonlocal found
        found = True
        return "[BANK_ACCOUNT:***]"

    text = BANK_ACCOUNT_PATTERN.sub(_bank_repl, text)

    return found, text


def check_output(llm_output: str) -> Tuple[bool, str]:
    """Verifica se a saída do LLM vaza dados sensíveis (PII) e os mascara.

    Retorna (is_safe, output). Se PII for detectado, is_safe=False e output contém o texto mascarado.
    """
    found, safe_output = _mask_pii(llm_output)
    if found:
        logger.warning("Guardrail de Output ativado: Vazamento de PII detectado e mascarado.")
        return False, safe_output

    return True, llm_output