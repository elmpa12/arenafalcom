#!/usr/bin/env python3
"""
Exemplo de uso do OpenAI Gateway
"""
import requests
import json

# URL do gateway
GATEWAY_URL = "https://bs3.falcomlabs.com/codex"

def list_models():
    """Lista todos os modelos disponíveis"""
    response = requests.get(f"{GATEWAY_URL}/api/models")
    models = response.json()["models"]
    print(f"📋 Modelos disponíveis ({len(models)}):")
    for model in models[:10]:  # Mostrar primeiros 10
        print(f"  - {model}")
    if len(models) > 10:
        print(f"  ... e mais {len(models) - 10} modelos")
    return models

def generate_code(prompt, model="gpt-4o"):
    """Gera código usando o gateway"""
    print(f"\n🤖 Gerando código com {model}...")
    print(f"📝 Prompt: {prompt}\n")
    
    response = requests.post(
        f"{GATEWAY_URL}/api/codex",
        json={"prompt": prompt, "model": model},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()["result"]
        print("✅ Código gerado:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        return result
    else:
        print(f"❌ Erro {response.status_code}: {response.text}")
        return None

def test_mock():
    """Testa com modelo mock (não gasta créditos)"""
    print("\n🧪 Testando modelo MOCK (grátis)...")
    result = generate_code("qualquer prompt", model="mock")
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 OpenAI Gateway - Exemplos de Uso")
    print("=" * 60)
    
    # 1. Listar modelos
    models = list_models()
    
    # 2. Testar com mock
    test_mock()
    
    # 3. Exemplos reais
    exemplos = [
        "criar uma função para calcular fatorial em python",
        "criar uma função async em javascript para fazer fetch de API",
        "criar uma classe Stack em python com push, pop e peek"
    ]
    
    print("\n" + "=" * 60)
    print("📚 Exemplos de Geração de Código")
    print("=" * 60)
    
    for i, prompt in enumerate(exemplos, 1):
        print(f"\n{'='*60}")
        print(f"Exemplo {i}/{len(exemplos)}")
        print('='*60)
        generate_code(prompt)
        
        if i < len(exemplos):
            input("\nPressione ENTER para próximo exemplo...")
    
    print("\n✨ Fim dos exemplos!")
    print("\n💡 Dica: Edite este script e adicione seus próprios prompts!")
