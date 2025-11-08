# 🌐 Guia HTTPie - Datalid API

Exemplos práticos de como usar o HTTPie para testar a API.

## 📦 Instalação

```bash
pip install httpie
```

## 🚀 Exemplos Básicos

### 1. Health Check
```bash
http GET localhost:8000/health
```

### 2. Informações da API
```bash
http GET localhost:8000/info
```

### 3. Documentação
Abra no navegador: http://localhost:8000/docs

---

## 📸 Processamento de Imagem

### Sintaxe Básica
```bash
http -f POST localhost:8000/process file@caminho/para/imagem.jpg
```

**Importante:** 
- Use `-f` para indicar multipart/form-data
- O nome do campo é `file` (não `image`)
- Use `@` antes do caminho do arquivo

### Exemplo Simples
```bash
http -f POST localhost:8000/process file@data/sample.jpg
```

### Com Parâmetros Personalizados
```bash
http -f POST localhost:8000/process \
  file@data/sample.jpg \
  confidence=0.5 \
  iou_threshold=0.7 \
  save_visualization=true
```

### Com Caminhos Absolutos (Windows)
```bash
http -f POST localhost:8000/process file@C:/Users/rafael/Pictures/produto.jpg
```

### Ver Headers e Body
```bash
http -f POST localhost:8000/process \
  file@data/sample.jpg \
  --print=hb
```

Opções de `--print`:
- `h` - Headers
- `b` - Body
- `hb` - Headers + Body
- `HBhb` - Tudo (request + response)

---

## 📦 Processamento em Batch

### Múltiplas Imagens
```bash
http -f POST localhost:8000/process/batch \
  files@image1.jpg \
  files@image2.jpg \
  files@image3.jpg
```

### Batch com Configuração
```bash
http -f POST localhost:8000/process/batch \
  files@img1.jpg \
  files@img2.jpg \
  confidence=0.6 \
  save_crops=true
```

---

## 🔧 Parâmetros Disponíveis

### Detecção (YOLO)
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  confidence=0.25 \           # 0.0 - 1.0
  iou_threshold=0.7 \         # 0.0 - 1.0
  model_name="yolov8m"        # nome do modelo
```

### OCR
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  ocr_engine="openocr" \      # openocr, tesseract, easyocr, etc.
  ocr_languages="['por','eng']"
```

### Visualização
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  save_visualization=true \
  save_crops=true \
  return_base64=false
```

---

## 🔐 Autenticação (se habilitada)

### Com API Key
```bash
http -f POST localhost:8000/process \
  "X-API-Key:sua-chave-aqui" \
  file@imagem.jpg
```

### Com Bearer Token
```bash
http -f POST localhost:8000/process \
  "Authorization:Bearer seu-token-jwt" \
  file@imagem.jpg
```

---

## 📊 Exemplos Avançados

### 1. Processar e Salvar Resultados
```bash
http -f POST localhost:8000/process \
  file@produto.jpg \
  save_visualization=true \
  save_crops=true \
  confidence=0.7 \
  > resultado.json
```

### 2. Pipeline Completo
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  confidence=0.6 \
  iou_threshold=0.5 \
  ocr_engine="openocr" \
  save_visualization=true \
  return_base64=false
```

### 3. Teste de Performance
```bash
time http -f POST localhost:8000/process file@imagem.jpg
```

### 4. Ver Apenas Status Code
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  --print=h | grep HTTP
```

### 5. Pretty Print JSON
```bash
http -f POST localhost:8000/process file@imagem.jpg | jq .
```

---

## 🎯 Casos de Uso Práticos

### Testar Alta Confiança
```bash
http -f POST localhost:8000/process \
  file@validade.jpg \
  confidence=0.8
```

### Processar Múltiplos Produtos
```bash
for img in produtos/*.jpg; do
  echo "Processando: $img"
  http -f POST localhost:8000/process file@"$img"
done
```

### Salvar Visualizações
```bash
http -f POST localhost:8000/process \
  file@produto.jpg \
  save_visualization=true \
  --output=resultado.json

# A visualização será salva em outputs/api_results/
```

---

## ❌ Erros Comuns

### ❌ Erro: "Field required"
**Problema:** Nome do campo incorreto
```bash
# ❌ Errado
http -f POST localhost:8000/process image@foto.jpg

# ✅ Correto
http -f POST localhost:8000/process file@foto.jpg
```

### ❌ Erro: "File not found"
**Problema:** Caminho do arquivo incorreto
```bash
# ✅ Use caminhos relativos ou absolutos corretos
http -f POST localhost:8000/process file@./data/sample.jpg
http -f POST localhost:8000/process file@C:/Users/rafael/imagem.jpg
```

### ❌ Erro: "Unsupported media type"
**Problema:** Faltou o `-f` para multipart/form-data
```bash
# ❌ Errado
http POST localhost:8000/process file@foto.jpg

# ✅ Correto
http -f POST localhost:8000/process file@foto.jpg
```

### ❌ Erro: "Connection refused"
**Problema:** API não está rodando
```bash
# Inicie a API primeiro
python -m src.api.main
```

---

## 🔍 Debug

### Ver Requisição Completa
```bash
http -v -f POST localhost:8000/process file@imagem.jpg
```

### Ver Apenas Response
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  --print=b
```

### Salvar Response
```bash
http -f POST localhost:8000/process \
  file@imagem.jpg \
  > resposta.json
```

### Ver Tempo de Resposta
```bash
time http -f POST localhost:8000/process file@imagem.jpg
```

---

## 📝 Respostas da API

### ✅ Sucesso (200)
```json
{
  "dates": [
    {
      "confidence": 0.85,
      "date_string": "12/2025",
      "days_until_expiry": 45,
      "formatted_date": "12/2025",
      "is_expired": false,
      "parsed_date": "2025-12-01T00:00:00"
    }
  ],
  "detection_count": 1,
  "image_size": [640, 480],
  "processed_at": "2025-11-05T03:30:00",
  "processing_time": 1.234,
  "status": "success"
}
```

### ❌ Erro de Validação (422)
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Erro de validação",
    "details": {
      "errors": [...]
    }
  },
  "timestamp": "2025-11-05T03:30:00"
}
```

### ❌ Erro Interno (500)
```json
{
  "status": "error",
  "error": {
    "code": "INTERNAL_SERVER_ERROR",
    "message": "Erro interno do servidor"
  },
  "timestamp": "2025-11-05T03:30:00"
}
```

---

## 🎓 Dicas

1. **Use tab completion:** HTTPie oferece autocomplete
2. **Teste incrementalmente:** Comece simples, adicione parâmetros
3. **Salve comandos úteis:** Crie aliases no shell
4. **Use jq para filtrar:** Combine com jq para extrair dados específicos
5. **Documente seus testes:** Salve os comandos que funcionaram

---

## 🔗 Links Úteis

- **HTTPie Docs:** https://httpie.io/docs
- **API Docs (Swagger):** http://localhost:8000/docs
- **API Docs (ReDoc):** http://localhost:8000/redoc

---

**Última atualização:** Novembro 2025
