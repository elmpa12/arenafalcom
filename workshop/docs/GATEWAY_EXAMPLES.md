# Como Usar o OpenAI Gateway

## 🌐 URL Base
```
https://bs3.falcomlabs.com/codex
```

## 📋 Endpoints Disponíveis

### 1. Health Check
Verificar se o serviço está ativo:
```bash
curl https://bs3.falcomlabs.com/codex/health
```
Resposta:
```json
{"status":"ok"}
```

### 2. Listar Modelos Disponíveis
```bash
curl https://bs3.falcomlabs.com/codex/api/models
```
Resposta:
```json
{
  "models": ["mock", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", ...]
}
```

### 3. Gerar Código

#### Exemplo 1: Usando modelo MOCK (não gasta créditos)
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "qualquer coisa",
    "model": "mock"
  }'
```
Resposta:
```json
{"result":"print('hello world')\n"}
```

#### Exemplo 2: Gerar código Python com GPT-4o
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "crie uma função recursiva para calcular fibonacci em python"
  }'
```

#### Exemplo 3: Gerar código JavaScript
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "crie uma função para ordenar array de objetos por nome em javascript"
  }'
```

#### Exemplo 4: Especificar modelo diferente
```bash
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "criar uma API REST em FastAPI",
    "model": "gpt-4-turbo"
  }'
```

## 🐍 Usando com Python

```python
import requests

url = "https://bs3.falcomlabs.com/codex/api/codex"

# Exemplo básico
response = requests.post(url, json={
    "prompt": "criar uma classe User com nome e email em python"
})

result = response.json()
print(result["result"])

# Com modelo específico
response = requests.post(url, json={
    "prompt": "algoritmo de busca binária",
    "model": "gpt-4o"
})

print(response.json()["result"])
```

## 🟨 Usando com JavaScript/Node.js

```javascript
const axios = require('axios');

async function generateCode(prompt, model = 'gpt-4o') {
  const response = await axios.post('https://bs3.falcomlabs.com/codex/api/codex', {
    prompt: prompt,
    model: model
  });
  
  return response.data.result;
}

// Uso
generateCode('criar uma função para validar email em javascript')
  .then(code => console.log(code))
  .catch(err => console.error(err));
```

## 🌐 Usando com Fetch (Browser)

```javascript
async function askCodex(prompt) {
  const response = await fetch('https://bs3.falcomlabs.com/codex/api/codex', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: prompt
    })
  });
  
  const data = await response.json();
  return data.result;
}

// Uso
askCodex('criar um componente React de contador')
  .then(code => console.log(code));
```

## 🔐 Se você configurou GATEWAY_TOKEN (autenticação)

```bash
# Adicione o header Authorization
curl -X POST https://bs3.falcomlabs.com/codex/api/codex \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer SEU_TOKEN_AQUI' \
  -d '{
    "prompt": "criar função de hash MD5"
  }'
```

## 🎨 Interface Web Simples (HTML)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Code Generator</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; }
        textarea { width: 100%; height: 100px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🤖 AI Code Generator</h1>
    
    <label>Descreva o código que você precisa:</label>
    <textarea id="prompt" placeholder="Ex: criar uma função para validar CPF em python"></textarea>
    
    <button onclick="generateCode()">Gerar Código</button>
    
    <h3>Resultado:</h3>
    <pre id="result">Aguardando...</pre>

    <script>
        async function generateCode() {
            const prompt = document.getElementById('prompt').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.textContent = 'Gerando...';
            
            try {
                const response = await fetch('https://bs3.falcomlabs.com/codex/api/codex', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                
                const data = await response.json();
                resultDiv.textContent = data.result;
            } catch (error) {
                resultDiv.textContent = 'Erro: ' + error.message;
            }
        }
    </script>
</body>
</html>
```

## 📊 Documentação Interativa (Swagger/OpenAPI)

Acesse a documentação interativa em:
```
https://bs3.falcomlabs.com/codex/docs
```

## ⚙️ Modelos Disponíveis

- `mock` - Resposta fixa (teste sem gastar créditos)
- `gpt-4o` - Modelo padrão (recomendado)
- `gpt-4-turbo` - Mais rápido
- `gpt-3.5-turbo` - Mais barato
- Veja lista completa em `/api/models`

## 🚨 Códigos de Erro

- `400` - Modelo não suportado ou payload inválido
- `401` - Token de autenticação inválido (se configurado)
- `503` - Chave OpenAI não configurada
- `500` - Erro interno

## 💡 Dicas

1. **Use modelo mock para testes** - não gasta créditos
2. **Seja específico no prompt** - quanto mais detalhes, melhor o código
3. **Especifique a linguagem** - "em python", "em javascript", etc.
4. **Liste modelos primeiro** - para ver opções disponíveis
