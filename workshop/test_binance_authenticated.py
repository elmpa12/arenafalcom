#!/usr/bin/env python3
"""
Testa se API AUTENTICADA da Binance funciona (mesmo com IP bloqueado).

Às vezes Binance permite IP whitelisted com API keys válidas.
"""

import os
import sys

# Verificar se tem API keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("=" * 60)
    print("TESTE DE API AUTENTICADA - BINANCE")
    print("=" * 60)
    print()
    print("⚠️  Para testar API autenticada, você precisa:")
    print()
    print("1. Criar API keys na Binance (mesmo que testnet):")
    print("   https://www.binance.com/en/my/settings/api-management")
    print()
    print("2. Configurar variáveis de ambiente:")
    print("   export BINANCE_API_KEY='sua_key'")
    print("   export BINANCE_API_SECRET='sua_secret'")
    print()
    print("3. Rodar novamente este script")
    print()
    print("💡 IMPORTANTE:")
    print("   - Use API keys de TESTNET primeiro!")
    print("   - Adicione o IP deste servidor ao whitelist")
    print("   - Testnet: https://testnet.binancefuture.com/")
    print()
    sys.exit(0)

# Testar com keys
import hashlib
import hmac
import time
import requests

def test_authenticated_api():
    print("=" * 60)
    print("TESTANDO API AUTENTICADA - BINANCE FUTURES")
    print("=" * 60)
    print()

    base_url = "https://fapi.binance.com"

    # 1. Testar endpoint que requer autenticação
    endpoint = "/fapi/v1/account"
    timestamp = int(time.time() * 1000)

    params = f"timestamp={timestamp}"
    signature = hmac.new(
        api_secret.encode('utf-8'),
        params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    url = f"{base_url}{endpoint}?{params}&signature={signature}"
    headers = {
        'X-MBX-APIKEY': api_key
    }

    print(f"1️⃣  Testando Account Info (requer auth):")
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        print(f"   Status: {resp.status_code}")

        if resp.status_code == 200:
            print("   ✅ FUNCIONOU! API autenticada ACESSÍVEL!")
            print()
            data = resp.json()
            print(f"   Balance: {len(data.get('assets', []))} assets")
            print(f"   Can Trade: {data.get('canTrade', False)}")
            print()
            print("🎉 EXCELENTE! Você PODE usar Binance em produção!")
            print("   O bloqueio é apenas em endpoints públicos.")
            return True
        elif resp.status_code == 451:
            print("   ❌ Ainda bloqueado (erro 451)")
            print(f"   Resposta: {resp.text[:200]}")
            return False
        else:
            print(f"   ⚠️  Erro: {resp.status_code}")
            print(f"   Resposta: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False

if __name__ == "__main__":
    test_authenticated_api()
