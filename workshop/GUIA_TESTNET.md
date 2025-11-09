# Guia: Paper Trading com Binance Futures Testnet

## O que é Testnet?

Ambiente de testes da Binance com:
- ✅ Dinheiro FAKE (não é real)
- ✅ Preços REAIS (streaming do mainnet)
- ✅ Ordens REAIS (executam de verdade no testnet)
- ✅ Todos os pares (BTC, ETH, SOL, etc)
- ✅ Mesmo comportamento do mainnet

**Use para:** Testar estratégias sem risco!

---

## 1. Criar Conta no Testnet

**URL:** https://testnet.binancefuture.com/

1. Registrar email + senha
2. Fazer login
3. Ir em: **API Key Management**
4. Criar nova API key
5. Copiar:
   - API Key
   - Secret Key

**IMPORTANTE:** São keys DIFERENTES do mainnet!

---

## 2. Configurar no Projeto

Criar arquivo `.env.testnet`:

```bash
# Testnet Keys (dinheiro FAKE)
BINANCE_TESTNET_API_KEY=sua_key_testnet_aqui
BINANCE_TESTNET_API_SECRET=sua_secret_testnet_aqui

# Configuração
TESTNET=true
TESTNET_BASE_URL=https://testnet.binancefuture.com
```

---

## 3. Código Python para Testnet

```python
import os
from binance.client import Client

# Configuração Testnet
TESTNET = True
TESTNET_URL = "https://testnet.binancefuture.com"

api_key = os.getenv('BINANCE_TESTNET_API_KEY')
api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

# Cliente Testnet
client = Client(
    api_key,
    api_secret,
    testnet=True,  # IMPORTANTE!
    tld='com'
)

# Testar
print("✅ Conectado ao Testnet!")

# Ver saldo (fake money)
balance = client.futures_account_balance()
for b in balance:
    if float(b['balance']) > 0:
        print(f"{b['asset']}: {b['balance']}")

# Ver preço
ticker = client.futures_symbol_ticker(symbol='BTCUSDT')
print(f"\nBTC Price: ${ticker['price']}")

# Criar ordem LIMIT (exemplo)
# order = client.futures_create_order(
#     symbol='BTCUSDT',
#     side='BUY',
#     type='LIMIT',
#     timeInForce='GTC',
#     quantity=0.001,
#     price=50000  # preço limite
# )
```

---

## 4. Diferenças Testnet vs Mainnet

| Feature | Testnet | Mainnet |
|---------|---------|---------|
| Dinheiro | Fake | Real |
| Preços | Real (streaming) | Real |
| Latência | ~300-400ms | ~7-30ms |
| Liquidez | Menor | Total |
| Slippage | Maior | Real |
| API Keys | Separadas | Separadas |
| Risco | Zero | Total |

**Importante:**
- Latência maior no testnet (não serve para medir performance real)
- Liquidez artificial (pode ter slippage diferente)
- Use apenas para validar LÓGICA, não performance

---

## 5. Obter Saldo Fake

No testnet você começa com saldo fake. Se acabar:

1. Login em https://testnet.binancefuture.com/
2. Menu → **Faucet** ou **Get Test Funds**
3. Solicitar mais USDT fake

---

## 6. Endpoints Testnet

```python
# Futures Testnet
BASE_URL = "https://testnet.binancefuture.com"

# Endpoints importantes:
GET /fapi/v1/ping              # Testar conexão
GET /fapi/v1/time              # Server time
GET /fapi/v1/exchangeInfo      # Símbolos disponíveis
GET /fapi/v1/ticker/price      # Preços
GET /fapi/v1/depth             # Order book

# Autenticados (precisam API key):
GET  /fapi/v1/account          # Saldo
POST /fapi/v1/order            # Criar ordem
DELETE /fapi/v1/order          # Cancelar ordem
GET  /fapi/v1/openOrders       # Ordens abertas
GET  /fapi/v1/allOrders        # Histórico
```

---

## 7. WebSocket Testnet

```python
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

def message_handler(message):
    print(message)

# WebSocket Testnet
ws_client = UMFuturesWebsocketClient(
    on_message=message_handler,
    testnet=True
)

# Stream de preço
ws_client.agg_trade(
    symbol='btcusdt',
    id=1,
)

# Manter conectado
import time
time.sleep(60)

ws_client.stop()
```

---

## 8. Exemplo Completo - Ordem Market

```python
import os
from binance.client import Client

# Setup Testnet
api_key = os.getenv('BINANCE_TESTNET_API_KEY')
api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

client = Client(api_key, api_secret, testnet=True)

# Ver saldo antes
balance_before = client.futures_account_balance()
print("Saldo antes:")
for b in balance_before:
    if b['asset'] == 'USDT':
        print(f"  USDT: {b['balance']}")

# Criar ordem MARKET
print("\nCriando ordem BUY MARKET...")
order = client.futures_create_order(
    symbol='BTCUSDT',
    side='BUY',
    type='MARKET',
    quantity=0.001  # 0.001 BTC
)

print(f"✅ Ordem executada!")
print(f"  Order ID: {order['orderId']}")
print(f"  Price: ${order['avgPrice']}")
print(f"  Quantity: {order['executedQty']} BTC")

# Ver posição
positions = client.futures_position_information(symbol='BTCUSDT')
for pos in positions:
    if float(pos['positionAmt']) != 0:
        print(f"\nPosição aberta:")
        print(f"  Amount: {pos['positionAmt']} BTC")
        print(f"  Entry Price: ${pos['entryPrice']}")
        print(f"  PnL: ${pos['unRealizedProfit']}")
```

---

## 9. Limitações do Testnet

**Testnet é bom para:**
- ✅ Validar lógica de ordens
- ✅ Testar API integration
- ✅ Validar risk management
- ✅ Aprender a usar a API

**Testnet NÃO é bom para:**
- ❌ Medir latência real
- ❌ Testar slippage real
- ❌ Validar performance HFT
- ❌ Simular condições extremas

---

## 10. Quando Migrar para Mainnet?

Migre quando:
1. ✅ Backtests lucrativos (Sharpe > 2)
2. ✅ Walk-forward validation OK
3. ✅ Paper trading funcionando > 1 mês
4. ✅ Gestão de risco validada
5. ✅ Estratégia robusta em múltiplos regimes

**Então:**
- Contratar VPS Brasil ($8-12/mês)
- Migrar código
- Começar com $ pequeno
- Escalar gradualmente

---

## Referências

- Testnet: https://testnet.binancefuture.com/
- Docs: https://binance-docs.github.io/apidocs/futures/en/
- Python SDK: https://github.com/binance/binance-connector-python

---

**Próximo passo:** Baixar dados históricos e começar backtests!
