# Gateway OpenAI - Guia de Uso

## 🚀 Comandos Rápidos

### Usando o CLI `flabs`

```bash
# Gerar código (usa GPT-4o por padrão)
flabs "criar uma API REST com FastAPI"

# Usar modelo específico
flabs "hello world em Python" gpt-4o
flabs "teste rápido" mock

# Alias curto
ai "criar função de ordenação quicksort"
```

## 📡 API HTTP Direta

### Health Check
```bash
curl https://bs3.falcomlabs.com/codex/health
```

### Listar Modelos Disponíveis
```bash
curl https://bs3.falcomlabs.com/codex/api/models
```

### Gerar Código
```bash
# Modelo padrão (gpt-4o)
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"criar função fibonacci em python"}'

# Modelo específico
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello world","model":"gpt-4o-mini"}'

# Mock (resposta instantânea, sem usar API)
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"teste","model":"mock"}'
```

### Com Token de Autenticação (se configurado)
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer SEU_TOKEN' \
  -d '{"prompt":"seu prompt aqui"}'
```

## 🐍 Usando em Python

```python
import requests

def gerar_codigo(prompt, modelo="gpt-4o"):
    url = "https://bs3.falcomlabs.com/codex/api/codex"
    payload = {
        "prompt": prompt,
        "model": modelo
    }
    response = requests.post(url, json=payload)
    return response.json()["result"]

# Exemplo
codigo = gerar_codigo("criar função que valida CPF")
print(codigo)
```

## 🌐 Usando em JavaScript

```javascript
async function gerarCodigo(prompt, modelo = "gpt-4o") {
  const response = await fetch("https://bs3.falcomlabs.com/codex/api/codex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model: modelo })
  });
  const data = await response.json();
  return data.result;
}

// Exemplo
gerarCodigo("criar componente React de login").then(console.log);
```

## 📋 Modelos Disponíveis

Execute para ver lista completa atualizada:
```bash
flabs
# ou
curl https://bs3.falcomlabs.com/codex/api/models | jq
```

Principais modelos:
- `gpt-4o` - Modelo padrão, rápido e eficiente
- `gpt-4o-mini` - Versão menor, mais econômica
- `gpt-4-turbo` - Alta performance
- `gpt-3.5-turbo` - Rápido e econômico
- `mock` - Retorna hello world (teste sem consumir API)

## ⚙️ Gerenciar Servidor

```bash
# Ver logs
tail -f /tmp/gateway.log

# Verificar status
ps aux | grep uvicorn

# Reiniciar
pkill -f "uvicorn backend.openai_gateway"
cd /opt/botscalpv3
. ../.venv/bin/activate
nohup uvicorn backend.openai_gateway:app --host 0.0.0.0 --port 8000 > /tmp/gateway.log 2>&1 &
```

## 🔧 Configuração (.env)

```bash
# Obrigatório
OPENAI_API_KEY=sk-xxxx

# Opcional
GATEWAY_TOKEN=seu-token-secreto
GATEWAY_ROOT_PATH=/codex
GATEWAY_PUBLIC_URL=https://bs3.falcomlabs.com/codex
```

## 📝 Exemplos Práticos

### Debug de código
```bash
flabs "explicar este erro: TypeError: 'NoneType' object is not subscriptable"
```

### Documentação
```bash
flabs "criar docstring para função que calcula média ponderada"
```

### Refatoração
```bash
flabs "refatorar este código para usar list comprehension: for i in range(10): if i % 2 == 0: result.append(i)"
```

### Testes
```bash
flabs "criar testes pytest para função de validação de email"
```

## 🎯 Dicas

1. **Mock para testes rápidos**: Use `model=mock` para validar integração sem gastar créditos
2. **Escolha o modelo certo**: gpt-4o para qualidade, gpt-3.5-turbo para velocidade
3. **Seja específico**: Quanto mais contexto no prompt, melhor o resultado
4. **Use o CLI**: `flabs` é mais rápido que curl para uso interativo

## 🔗 Endpoints

- Base: `https://bs3.falcomlabs.com/codex`
- Health: `/health`
- Modelos: `/api/models`
- Codex: `/api/codex`
- Docs: `/docs` (Swagger UI)
- ReDoc: `/redoc` (Documentação alternativa)
