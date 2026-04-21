import os
import pandas as pd
import random
from faker import Faker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_financial_data(output_path: str, num_records: int = 100):
    """
    Gera dados sintéticos financeiros para o ambiente de desenvolvimento (Dev/Staging),
    garantindo que dados reais/PII não sejam expostos durante o desenvolvimento (GAP 08).
    """
    fake = Faker('pt_BR')
    data = []
    
    logger.info(f"Gerando {num_records} registros de dados financeiros sintéticos...")
    
    for _ in range(num_records):
        data.append({
            "id_transacao": fake.uuid4(),
            "data_registro": fake.date_between(start_date='-1y', end_date='today'),
            "ticker": random.choice(["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]),
            "volume_negociado": random.randint(1000, 500000),
            "preco_fechamento": round(random.uniform(10.0, 100.0), 2),
            "analista_responsavel": fake.name() # Nome fictício
        })
        
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Dados sintéticos gerados com sucesso em: {output_path}")

if __name__ == "__main__":
    # Salva na pasta de dados processados para uso local
    generate_synthetic_financial_data("data/processed/synthetic_data_dev.csv")