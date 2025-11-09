#!/usr/bin/env python3
"""
Exemplo de uso do Gateway OpenAI em Python
"""
import requests
import json

GATEWAY_URL = "https://bs3.falcomlabs.com/codex/api/codex"

def gerar_codigo(prompt, modelo="gpt-4o"):
    """Gera código usando o Gateway OpenAI"""
    payload = {
        "prompt": prompt,
        "model": modelo
    }
    
    response = requests.post(GATEWAY_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()["result"]
    else:
        return f"Erro {response.status_code}: {response.text}"

if __name__ == "__main__":
    # Exemplo 1: Criar função
    print("=" * 60)
    print("EXEMPLO 1: Criar função de validação")
    print("=" * 60)
    codigo = gerar_codigo("criar função Python para validar CPF")
    print(codigo)
    
    # Exemplo 2: Mock rápido
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Teste com mock (instantâneo)")
    print("=" * 60)
    mock = gerar_codigo("qualquer coisa", modelo="mock")
    print(mock)
    
    # Exemplo 3: Explicar código
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Explicar erro")
    print("=" * 60)
    explicacao = gerar_codigo(
        "explicar e corrigir: list index out of range quando acesso array[10] em array de 5 elementos"
    )
    print(explicacao)
