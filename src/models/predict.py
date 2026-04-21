import torch
import joblib
import numpy as np
import json
import os
from src.models.model import LSTMModel
from src.config import MODEL_PATH, SCALER_PATH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tenta carregar configuração dinâmica do melhor modelo
        config_path = os.path.join(os.path.dirname(MODEL_PATH), "model_hyperparameters.json")
        
        h_size = HIDDEN_SIZE
        n_layers = NUM_LAYERS
        drop = DROPOUT
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    h_size = config.get("hidden_size", HIDDEN_SIZE)
                    n_layers = config.get("num_layers", NUM_LAYERS)
                    drop = config.get("dropout", DROPOUT)
                print(f"Carregando modelo com arquitetura dinâmica: {config}")
            except Exception as e:
                print(f"Erro ao ler config dinâmica: {e}. Usando defaults.")
        
        self.model = LSTMModel(hidden_size=h_size, num_layers=n_layers, dropout=drop)
        
        # Carrega pesos
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Carrega Scaler
        self.scaler = joblib.load(SCALER_PATH)
        
    def predict_next_day(self, last_60_days_prices):
        # Prepara input
        input_data = np.array(last_60_days_prices).reshape(-1, 1)
        scaled_input = self.scaler.transform(input_data)
        
        # Converte para tensor (Batch, Seq, Feature) -> (1, 60, 1)
        X = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X)
            
        # Desnormaliza
        prediction_price = self.scaler.inverse_transform(prediction.cpu().numpy())
        return float(prediction_price[0][0])