import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carrega a variável GOOGLE_API_KEY do arquivo .env
load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Modelos disponíveis para Geração de Texto na sua conta:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        # Imprime apenas o nome final que você deve usar no .env
        print(m.name.replace('models/', ''))
