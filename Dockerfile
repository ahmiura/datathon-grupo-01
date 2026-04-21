FROM python:3.10-slim

WORKDIR /app

# 1. Copia o arquivo de gerenciamento de dependências
COPY pyproject.toml .

# Copia o código fonte (Necessário para o build do Poetry/PEP-517)
COPY . .

# 2. Instala as dependências garantindo a arquitetura correta
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

# Cria diretório de modelos (caso não esteja montado como volume)
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]