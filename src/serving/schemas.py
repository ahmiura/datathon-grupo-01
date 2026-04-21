from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    last_60_days: Optional[List[float]] = Field(default=None, min_items=60, max_items=60, description="Lista opcional. Se vazio, busca dados do Yahoo Finance.")

class PredictionResponse(BaseModel):
    ticker: str
    predicted_price: float