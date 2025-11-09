#!/usr/bin/env python3
"""
Test Anthropic API - Verificar quais modelos estão disponíveis
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key (primeiros 20 chars): {api_key[:20]}...")
print(f"API Key (últimos 10 chars): ...{api_key[-10:]}")

client = Anthropic(api_key=api_key)

# Tentar diferentes modelos
models_to_test = [
    "claude-3-5-sonnet-20241022",  # Mais recente
    "claude-3-5-sonnet-20240620",  # Junho 2024
    "claude-3-opus-20240229",      # Opus (mais poderoso)
    "claude-3-sonnet-20240229",    # Atual (deprecated)
    "claude-3-haiku-20240307",     # Haiku (mais rápido)
]

print("\n" + "="*60)
print("TESTANDO MODELOS DISPONÍVEIS")
print("="*60)

for model in models_to_test:
    try:
        print(f"\n🧪 Testando: {model}")
        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'API working!'"}]
        )
        print(f"✅ FUNCIONA! Resposta: {response.content[0].text}")
        break  # Se funcionar, para aqui
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not_found" in error_str:
            print(f"❌ Modelo não encontrado")
        elif "401" in error_str or "authentication" in error_str.lower():
            print(f"❌ Erro de autenticação: {error_str[:100]}")
        else:
            print(f"❌ Erro: {error_str[:100]}")

print("\n" + "="*60)
