# 🚀 Guia de Início Rápido

> Comece a usar o Datalid 3.0 em menos de 5 minutos!

## ⚡ Setup Ultra-Rápido

### 1. Clone e Instale (2 minutos)

```bash
# Clone o repositório
git clone [seu-repo]
cd datalid3.0

# Instale as dependências
pip install -r requirements.txt

# Teste a instalação
make validate-env
```

### 2. Primeiro Teste (1 minuto)

```bash
# Teste em uma imagem de exemplo
make pipeline-test IMAGE=data/ocr_test/sample.jpg
```

**Pronto!** 🎉 Você acabou de detectar e extrair uma data de validade!

## 📸 Teste com Sua Própria Imagem

```bash
# Substitua pelo caminho da sua imagem
make pipeline-test IMAGE=/caminho/para/sua/imagem.jpg
```

### Resultado Esperado

```
🚀 Processando imagem: sua_imagem.jpg
📍 [1/3] Executando detecção YOLO...
✅ 1 região(ões) detectada(s)
🔍 [2/3] Executando OCR nas regiões...
✅ Texto extraído: "VAL: 15/03/2025"
📅 [3/3] Fazendo parse de datas...
✅ Data encontrada: 15/03/2025 (confiança: 0.95)
✅ Pipeline concluído em 1.23s

📊 RESULTADO FINAL:
   Data: 15/03/2025
   Confiança: 95%
   Formato: DD/MM/YYYY
```

## 🎯 Casos de Uso Rápidos

### Caso 1: Processar Múltiplas Imagens

```bash
# Avaliar 10 imagens aleatórias
make pipeline-eval-quick

# Avaliar 20 imagens específicas
make pipeline-eval NUM=20 MODE=random
```

### Caso 2: Usar API REST

```bash
# 1. Inicie a API
make api-run

# 2. Em outro terminal, teste
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg"
```

### Caso 3: Cliente Python

```python
from scripts.api.client import DatalidClient

# Conecte à API
client = DatalidClient("http://localhost:8000")

# Processe uma imagem
result = client.process_image("produto.jpg")

# Acesse a data extraída
print(f"Data: {result['best_date']['date']}")
print(f"Confiança: {result['best_date']['confidence']}")
```

## 🔧 Comandos Make Essenciais

```bash
# TESTES BÁSICOS
make pipeline-test IMAGE=imagem.jpg    # Testar em uma imagem
make pipeline-eval-quick               # Avaliação rápida (10 imgs)
make pipeline-eval-full                # Avaliação completa

# API
make api-run                          # Iniciar API
make api-test                         # Testar API
make api-health                       # Verificar status

# VALIDAÇÃO
make validate-env                     # Verificar ambiente
make test-cuda                        # Testar GPU/CUDA

# VISUALIZAÇÃO
make tensorboard                      # Ver métricas de treino
```

## 📊 Entendendo os Resultados

### Estrutura do Output

```json
{
  "success": true,
  "best_date": {
    "date": "15/03/2025",
    "confidence": 0.95,
    "format": "DD/MM/YYYY",
    "text": "VAL: 15/03/2025"
  },
  "detections": [
    {
      "bbox": [120, 80, 450, 120],
      "confidence": 0.87,
      "has_mask": true
    }
  ],
  "processing_time": 1.23
}
```

### Métricas de Confiança

| Score | Interpretação | Ação Recomendada |
|-------|--------------|------------------|
| 0.9 - 1.0 | Excelente | Usar diretamente |
| 0.7 - 0.9 | Boa | Usar com confiança |
| 0.5 - 0.7 | Moderada | Revisar manualmente |
| < 0.5 | Baixa | Verificar imagem/qualidade |

## 🎨 Visualizações Geradas

Após processar uma imagem, você encontrará:

```
outputs/
└── pipeline/
    └── sua_imagem/
        ├── result.json              # Resultado completo
        ├── annotated.jpg            # Imagem com anotações
        └── crops/
            └── crop_0.jpg           # Região detectada
```

## ⚙️ Configuração Básica

### Trocar Modelo YOLO

Edite `config/pipeline/full_pipeline.yaml`:

```yaml
detection:
  model_path: experiments/yolov8s_seg/weights/best.pt  # small
  # ou
  model_path: experiments/yolov8m_seg/weights/best.pt  # medium (padrão)
  # ou  
  model_path: experiments/yolov8n_seg/weights/best.pt  # nano (mais rápido)
```

### Trocar Engine OCR

```yaml
ocr:
  engine: openocr        # Padrão (recomendado)
  # ou
  engine: parseq_enhanced  # PARSeq melhorado
  # ou
  engine: trocr           # TrOCR (transformer)
```

### Ajustar Confidence

```yaml
detection:
  confidence: 0.25  # Padrão (menos restritivo)
  # ou
  confidence: 0.5   # Mais restritivo (menos falsos positivos)
```

## 🐛 Problemas Comuns

### Erro: "No detections found"

**Causa**: Confiança muito alta ou imagem difícil  
**Solução**: Reduza `confidence` no config

```yaml
detection:
  confidence: 0.15  # Tente um valor menor
```

### Erro: "CUDA not available"

**Causa**: GPU não detectada  
**Solução**: Force CPU

```yaml
detection:
  device: cpu  # Em vez de 0 (GPU)
```

### OCR retorna texto incorreto

**Causa**: Pré-processamento inadequado  
**Solução**: Teste outro engine ou ajuste preprocessing

```bash
# Tente outro engine
make ocr-test ENGINE=parseq_enhanced

# Ou compare todos
make ocr-compare
```

## 📚 Próximos Passos

Agora que você testou o básico:

1. **[Entenda a Arquitetura](04-ARCHITECTURE.md)** - Como tudo funciona
2. **[Configure para seu caso](19-YAML-CONFIG.md)** - Ajuste fino
3. **[Use a API](16-API-REST.md)** - Integre em sua aplicação
4. **[Treine seu modelo](13-YOLO-TRAINING.md)** - Customize para seus dados

## 🎯 Benchmark de Performance

### Velocidade (GPU RTX 3060)

| Modelo | Tempo/Imagem | FPS | Precisão |
|--------|--------------|-----|----------|
| YOLOv8n-seg | 0.3s | ~3.3 | 85% |
| YOLOv8s-seg | 0.5s | ~2.0 | 90% |
| YOLOv8m-seg | 0.8s | ~1.2 | 93% |

## 💡 Dicas Rápidas

✅ **Use OpenOCR** para melhor precisão  
✅ **Use YOLOv8m-seg** para melhor detecção  
✅ **Reduza confidence** se não detectar nada  
✅ **Compare engines** com `make ocr-compare`  
✅ **Avalie regularmente** com `make pipeline-eval-quick`  

## 🎓 Tutoriais em Vídeo

*(Em construção - adicione links para tutoriais em vídeo)*

## ❓ Precisa de Ajuda?

- 📖 Veja o [FAQ](25-FAQ.md)
- 🐛 Consulte [Troubleshooting](22-TROUBLESHOOTING.md)
- 💬 Abra uma Issue no GitHub

---

**Próximo: [Instalação Completa →](02-INSTALLATION.md)**
