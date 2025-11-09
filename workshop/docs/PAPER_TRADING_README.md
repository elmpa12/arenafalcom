# 🚀 PAPER TRADING - BotScalp v3

Sistema completo de **Paper Trading** integrando IA Dual (Claude + GPT) com Binance Testnet.

## ✨ O que é isso?

Depois de 6 meses de desenvolvimento, **FINALMENTE** chegamos no paper trading! 🎉

Este sistema combina:
- **Claudex 2.0**: Claude + GPT debatem CADA trade antes de executar
- **Competitive Trader**: Análise de mercado e propostas de trade
- **Binance Testnet**: Execução REAL de ordens (sem risco, dinheiro fake)
- **Memória Persistente**: Sistema aprende com cada trade

## 📋 Pré-requisitos

1. **Credenciais Binance** no `.env`:
   ```bash
   BINANCE_API_KEY=sua_key_aqui
   BINANCE_API_SECRET=seu_secret_aqui
   ```

2. **Python 3.11+** com bibliotecas:
   ```bash
   pip install python-binance python-dotenv openai anthropic
   ```

## 🎯 Como Usar

### Teste Básico (Conexão)
```bash
# Testa apenas conexão com Binance Testnet
python3 paper_trading_executor.py
```

### Paper Trading Completo
```bash
# Executa 3 trades com debate Claude + GPT
python3 run_paper_trading.py --trades 3

# Executa 5 trades em ETHUSDT
python3 run_paper_trading.py --symbol ETHUSDT --trades 5

# Executa SEM debate (mais rápido, menos seguro)
python3 run_paper_trading.py --trades 3 --no-debate
```

## 🔄 Fluxo de Execução

Cada trade passa por:

```
1. 📊 Análise de Mercado
   └─> Preço atual, saldo, indicadores técnicos

2. 🧠 Proposta de Trade (Claude + GPT análise)
   └─> Ação (BUY/SELL), confiança, lógica de entrada

3. 💬 Debate (GPT-Strategist vs GPT-Executor)
   └─> EXECUTAR ou REJEITAR com justificativa

4. ✅ Confirmação Manual
   └─> Usuário confirma execução (safety)

5. ⚡ Execução na Exchange
   └─> Ordem real na Binance Testnet

6. 📝 Registro em Memória
   └─> Sistema aprende para próximos trades
```

## 📊 Exemplo de Saída

```
======================================================================
🎯 TRADE #1
======================================================================

📊 Analisando mercado para BTCUSDT...
   Preço atual: $95,234.50
   Saldo disponível: $10,000.00 USDT

🧠 Gerando proposta de trade (Claude + GPT)...

📋 PROPOSTA:
   Ação: BUY
   Símbolo: BTCUSDT
   Lógica: RSI oversold + MACD bullish cross
   Confiança: 85.0%

💬 Iniciando debate GPT-Strategist vs GPT-Executor...

======================================================================
🧠 GPT-STRATEGIST (abertura)...
======================================================================

💬 STRATEGIST:
┌────────────────────────────────────────────────────────────────────┐
│ Considerando o RSI oversold em território de 32 e o MACD           │
│ mostrando divergência bullish, há um setup técnico favorável.      │
│ Porém, devemos considerar o contexto macro...                      │
└────────────────────────────────────────────────────────────────────┘

[...]

💡 DECISÃO: ✅ EXECUTAR
   Raciocínio: Setup técnico sólido com confluência de 3 indicadores...

⚡ Executando trade na exchange...
   Investindo: $100.00 USDT
   Quantidade: 0.00105 BTC

⚠️  CONFIRMAR EXECUÇÃO REAL NO TESTNET? (y/n)
   > y

📤 Colocando ordem: BUY 0.00105 BTCUSDT
✅ Ordem executada! ID: 123456789
   Status: FILLED
   Filled: 0.00105 @ avg price market

✅ TRADE EXECUTADO COM SUCESSO!
   Order ID: 123456789
   Status: FILLED
```

## ⚙️ Configurações

### Símbolos Suportados
- `BTCUSDT` (padrão)
- `ETHUSDT`
- `BNBUSDT`
- Qualquer par da Binance Testnet

### Tamanho de Posição
- Usa **10% do saldo** disponível
- Máximo **$100 USDT** por trade
- Configurável em `run_paper_trading.py`

### Debate
- **Habilitado**: Claude + GPT debatem cada decisão (mais lento, mais seguro)
- **Desabilitado**: Executa propostas automaticamente (mais rápido)

## 🔐 Segurança

1. ✅ **Testnet por padrão**: Nunca usa dinheiro real
2. ✅ **Confirmação manual**: Sempre pede confirmação antes de executar
3. ✅ **Logs completos**: Todas as decisões registradas
4. ✅ **Rate limiting**: Pausa entre trades para evitar overtrading

## 📈 Próximos Passos

Depois que funcionar no testnet:

- [ ] Integrar com `selector21.py` (Walk-Forward otimização)
- [ ] Adicionar DL remoto (GPU predictions)
- [ ] Implementar ATR Stop/Takeprofit
- [ ] Visual Replay (replay de trades)
- [ ] Produção (⚠️ **só depois de 100+ trades lucrativos no testnet!**)

## 🐛 Troubleshooting

### Erro: "Invalid API-key"
→ Verifique se as keys da Binance Testnet estão corretas no `.env`

### Erro: "Insufficient balance"
→ Sua conta testnet precisa de saldo. Obtenha em: https://testnet.binance.vision/

### Erro: "Module not found: binance"
→ Instale: `pip3 install python-binance`

### SSL Certificate Error
→ Normal com GPT às vezes. Sistema continua funcionando.

## 📞 Suporte

Se algo não funcionar:
1. Verifique o `.env` com as credenciais
2. Confirme que está usando **Binance Testnet** (não produção!)
3. Veja os logs em `paper_trading_session_*.json`

---

**6 meses de trabalho culminam AQUI! 🎉**

Vamos fazer história nessa competição! 🚀
