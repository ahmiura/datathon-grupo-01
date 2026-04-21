import os

# Configurações de Dados
TICKER = "PETR4.SA"
START_DATE = "2021-01-01"
END_DATE = "2026-01-31"
TEST_START_DATE = "2025-01-01"

# Hiperparâmetros do Modelo
TIMESTEPS = 60       # Janela de 60 dias
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 50
NUM_LAYERS = 1
DROPOUT = 0.2

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")