FROM python:3.10-slim

WORKDIR /app

# 1. Copia o arquivo de gerenciamento de dependências
COPY pyproject.toml .

# Cria um diretório fake temporário para instalar as dependências em cache
RUN mkdir -p src && touch src/__init__.py && touch README.md

# 2. Instala as dependências garantindo a arquitetura correta
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

# 3. Copia o código fonte real (Mudanças no código agora NÃO reinstalam as libs!)
COPY . .

# Cria diretório de modelos (caso não esteja montado como volume)
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]