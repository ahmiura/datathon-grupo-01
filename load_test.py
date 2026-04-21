"""
Script de Teste de Carga e Geração de Tráfego.
Responsável por simular requisições HTTP para a API com o objetivo de
alimentar o Prometheus e visualizar as métricas no dashboard do Grafana.
"""

import requests
import time
import random

# URL da API
url = "http://localhost:8000/predict"

print("🚀 Iniciando Simulação de Tráfego...")
print("Pressione CTRL+C para parar.")

while True:
    try:
        # 50% de chance de enviar dados, 50% de pedir busca automática (que é mais lenta)
        if random.random() > 0.5:
            payload = {} 
            type_req = "AUTO (Yahoo)"
        else:
            payload = {"last_60_days": [random.uniform(20.0, 40.0) for _ in range(60)]}
            type_req = "MANUAL (Dados)"

        start_time = time.time()
        response = requests.post(url, json=payload)
        latency = time.time() - start_time

        if response.status_code == 200:
            print(f"✅ [{type_req}] Status: 200 | Latência: {latency:.4f}s")
        else:
            print(f"❌ [{type_req}] Status: {response.status_code} | Erro: {response.text}")

    except Exception as e:
        print(f"⚠️ Erro de conexão: {e}")
    
    # Aguarda um pouco para variar a carga
    time.sleep(random.uniform(0.1, 0.5))