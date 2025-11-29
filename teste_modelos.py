import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configura a chave direto na SDK do Google (sem LangChain) para testar
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- Modelos Disponíveis para sua Chave ---")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Erro ao listar: {e}")