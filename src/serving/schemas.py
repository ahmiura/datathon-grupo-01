from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    last_60_days: Optional[List[float]] = Field(default=None, min_length=60, max_length=60, description="Lista opcional. Se vazio, busca dados do Yahoo Finance.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "last_60_days": [35.5] * 60
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    ticker: str = Field(description="Ticker do ativo analisado.")
    predicted_price: float = Field(description="Preço de fechamento predito pelo modelo LSTM.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "PETR4.SA",
                    "predicted_price": 35.12
                }
            ]
        }
    }