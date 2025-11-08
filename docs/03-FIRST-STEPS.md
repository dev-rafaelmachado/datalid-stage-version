# 👣 Primeiros Passos

> Seus primeiros testes com o Datalid 3.0

## 🎯 Objetivo

Neste guia você vai:
- ✅ Testar o pipeline completo em uma imagem de exemplo
- ✅ Processar suas próprias imagens
- ✅ Entender os resultados
- ✅ Experimentar diferentes configurações

**Tempo estimado:** 10-15 minutos

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter:
- [x] Ambiente instalado ([Instalação](02-INSTALLATION.md))
- [x] Validação bem-sucedida (`make validate-env`)
- [x] Imagens de produtos com datas de validade

## 🚀 Teste 1: Imagem de Exemplo

### Passo 1: Testar com Amostra Incluída

O projeto já vem com uma imagem de exemplo:

```bash
make pipeline-test IMAGE=data/ocr_test/sample.jpg
```

**Resultado esperado:**

```
🚀 Iniciando Pipeline Completo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📸 Imagem: data/ocr_test/sample.jpg
🔍 Modelo: yolov8n-seg.pt
📝 OCR: OpenOCR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️  ETAPA 1: Detecção YOLO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Detectado 1 região(ões)
   📍 Box: [120, 340, 285, 395]
   🎯 Confiança: 0.94
   ⏱️  Tempo: 0.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 ETAPA 2: OCR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Texto extraído: "VAL 15/03/2025"
   🎯 Confiança: 0.92
   ⏱️  Tempo: 0.8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 ETAPA 3: Parse de Data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Data extraída: 15/03/2025
   📅 Formato: DD/MM/YYYY
   ⏳ Dias até expirar: 127
   ⚠️  Status: Válido
   🎯 Confiança final: 0.93

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ RESULTADO FINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Data de Validade: 15/03/2025
🎯 Confiança: 93%
⏱️  Tempo Total: 1.2s
📁 Visualização: outputs/pipeline/sample_result.jpg

✅ Pipeline executado com sucesso!
```

### Passo 2: Visualizar Resultado

Abra a imagem de resultado gerada:

```bash
# Windows
start outputs/pipeline/sample_result.jpg

# Linux
xdg-open outputs/pipeline/sample_result.jpg

# macOS
open outputs/pipeline/sample_result.jpg
```

**O que você verá:**
- 🟢 Bounding box verde ao redor da região detectada
- 📊 Máscara de segmentação (polígono azul transparente)
- 📝 Texto extraído pelo OCR
- 📅 Data parsed com confiança
- ⏰ Dias até a expiração

## 📸 Teste 2: Sua Própria Imagem

### Passo 1: Preparar Imagem

Tire uma foto de um produto com data de validade visível:

**Dicas para melhor resultado:**
- ✅ Boa iluminação (evite sombras)
- ✅ Foco nítido na data
- ✅ Data bem legível
- ✅ Evite reflexos (flash pode prejudicar)
- ✅ Imagem com resolução mínima 640px

### Passo 2: Processar

```bash
# Via linha de comando
make pipeline-test IMAGE=/caminho/para/sua/imagem.jpg

# Ou via Python
python scripts/pipeline/run_full_pipeline.py \
  --image /caminho/para/sua/imagem.jpg \
  --config config/pipeline/full_pipeline.yaml
```

### Passo 3: Analisar Resultado

O resultado será salvo em `outputs/pipeline/`:

```
outputs/pipeline/
├── sua_imagem_result.jpg          # Visualização com anotações
├── sua_imagem_crop_0.jpg          # Região detectada (crop)
├── sua_imagem_preprocessed_0.jpg  # Imagem após pré-processamento
└── sua_imagem_results.json        # Resultados detalhados em JSON
```

#### Estrutura do JSON

```json
{
  "image_path": "/caminho/para/sua/imagem.jpg",
  "timestamp": "2024-11-08T14:30:00",
  "detections": [
    {
      "bbox": [120, 340, 285, 395],
      "confidence": 0.94,
      "mask": [[x1, y1], [x2, y2], ...],
      "class": 0
    }
  ],
  "ocr_results": [
    {
      "text": "VAL 15/03/2025",
      "confidence": 0.92,
      "engine": "openocr"
    }
  ],
  "parsed_dates": [
    {
      "date": "15/03/2025",
      "format": "DD/MM/YYYY",
      "confidence": 0.93,
      "days_until_expiry": 127,
      "is_expired": false
    }
  ],
  "best_date": {
    "date": "15/03/2025",
    "confidence": 0.93,
    "days_until_expiry": 127,
    "is_expired": false
  },
  "processing_time": {
    "detection": 0.3,
    "preprocessing": 0.1,
    "ocr": 0.8,
    "parsing": 0.02,
    "total": 1.22
  }
}
```

## 🔄 Teste 3: Diferentes Configurações

### Experimentar Outros OCR Engines

O sistema suporta 7 engines OCR diferentes. Vamos testar alguns:

#### OpenOCR (Padrão - Recomendado)

```bash
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --ocr-config config/ocr/openocr.yaml
```

#### PARSeq (Bom para fontes estilizadas)

```bash
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --ocr-config config/ocr/parseq.yaml
```

#### TrOCR (Melhor para texto manuscrito)

```bash
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --ocr-config config/ocr/trocr.yaml
```

#### Tesseract (Rápido, boa baseline)

```bash
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --ocr-config config/ocr/tesseract.yaml
```

### Comparar Todos os OCR Engines

Execute o benchmark de OCR para comparar todos:

```bash
python scripts/ocr/benchmark_ocr.py \
  --image sua_imagem.jpg \
  --output outputs/ocr_benchmarks/
```

### Ajustar Pré-processamento

Para imagens com baixa qualidade, experimente pré-processamentos mais agressivos:

```bash
# Pré-processamento mínimo (padrão)
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --preprocessing-config config/preprocessing/ppro-minimal.yaml

# Pré-processamento completo (para imagens difíceis)
python scripts/pipeline/run_full_pipeline.py \
  --image sua_imagem.jpg \
  --preprocessing-config config/preprocessing/ppro-easyocr.yaml
```

## 📊 Teste 4: Processar Múltiplas Imagens

### Processar Diretório

```bash
python scripts/pipeline/run_full_pipeline.py \
  --input-dir data/minhas_imagens/ \
  --output-dir outputs/resultados/ \
  --save-visualizations
```

### Via Python Script

```python
# process_batch.py
from pathlib import Path
from src.pipeline.full_pipeline import FullPipeline
from src.ocr.config import load_pipeline_config

# Carregar configuração
config = load_pipeline_config("config/pipeline/full_pipeline.yaml")

# Inicializar pipeline
pipeline = FullPipeline(config)

# Processar todas as imagens em um diretório
image_dir = Path("data/minhas_imagens")
results = []

for image_path in image_dir.glob("*.jpg"):
    print(f"\n🔄 Processando: {image_path.name}")
    
    result = pipeline.process(str(image_path))
    results.append(result)
    
    if result['best_date']:
        print(f"✅ Data: {result['best_date']['date']}")
        print(f"🎯 Confiança: {result['best_date']['confidence']:.2%}")
    else:
        print("❌ Nenhuma data encontrada")

print(f"\n✅ Processadas {len(results)} imagens")
```

Execute:
```bash
python process_batch.py
```

## 🔍 Teste 5: Entender Falhas

### Caso 1: Nenhuma Região Detectada

**Sintoma:**
```
❌ Nenhuma região detectada pela YOLO
```

**Possíveis causas e soluções:**

1. **Data muito pequena na imagem**
   ```yaml
   # config/pipeline/full_pipeline.yaml
   detection:
     imgsz: 1280  # Aumentar resolução (padrão: 640)
     conf: 0.15   # Reduzir threshold de confiança (padrão: 0.25)
   ```

2. **Data em posição/contexto incomum**
   - Use modelo treinado com mais dados
   - Considere retreinar YOLO com suas imagens

3. **Imagem de baixa qualidade**
   - Tire foto melhor (boa iluminação, foco)
   - Aumente resolução da imagem

### Caso 2: Texto Extraído Incorretamente

**Sintoma:**
```
❌ OCR extraiu: "YAL 15i03i2025" (incorreto)
✅ Correto seria: "VAL 15/03/2025"
```

**Soluções:**

1. **Experimentar outro OCR engine**
   ```bash
   # Testar PARSeq
   python scripts/pipeline/run_full_pipeline.py \
     --image sua_imagem.jpg \
     --ocr-config config/ocr/parseq.yaml
   ```

2. **Melhorar pré-processamento**
   ```yaml
   # config/preprocessing/ppro-custom.yaml
   preprocessing:
     enable: true
     clahe: true       # Melhorar contraste
     denoise: true     # Remover ruído
     sharpen: true     # Aumentar nitidez
     binarize: true    # Binarização adaptativa
   ```

3. **Ajustar crop da região**
   ```yaml
   detection:
     crop_padding: 15  # Aumentar margem do crop (padrão: 10)
   ```

### Caso 3: Data Não Parseada

**Sintoma:**
```
✅ OCR: "Validade 15 03 2025"
❌ Parse falhou: nenhum formato reconhecido
```

**Soluções:**

1. **Adicionar formatos customizados**
   ```yaml
   # config/pipeline/full_pipeline.yaml
   parsing:
     date_formats:
       - "%d/%m/%Y"
       - "%d-%m-%Y"
       - "%d.%m.%Y"
       - "%d %m %Y"      # Adicionar formato com espaços
       - "Validade %d %m %Y"  # Com prefixo
   ```

2. **Ativar fuzzy matching**
   ```yaml
   parsing:
     fuzzy_matching: true
     fuzzy_threshold: 0.7
   ```

## 📈 Próximos Passos

Agora que você já testou o sistema:

1. **Entender a Teoria** → [Teoria e Conceitos](05-THEORY.md)
2. **Conhecer a Arquitetura** → [Arquitetura do Sistema](04-ARCHITECTURE.md)
3. **Explorar Componentes** → [Sistema OCR](08-OCR-SYSTEM.md)
4. **Integrar em Aplicação** → [API REST](16-API-REST.md)
5. **Treinar Modelo Próprio** → [Treinamento YOLO](13-YOLO-TRAINING.md)

## 💡 Dicas

- 📸 **Qualidade da imagem é fundamental** - boa iluminação e foco
- 🔄 **Experimente diferentes OCR engines** - cada um tem pontos fortes
- ⚙️ **Ajuste configurações** quando necessário
- 📊 **Analise os JSONs** para entender resultados detalhados
- 🐛 **Use visualizações** para debugar problemas

## 🆘 Precisa de Ajuda?

- 📚 [FAQ](25-FAQ.md) - Perguntas frequentes
- 🐛 [Troubleshooting](22-TROUBLESHOOTING.md) - Resolução de problemas
- 💬 Abra uma issue no GitHub

---

**Parabéns!** 🎉 Você completou seus primeiros passos com o Datalid 3.0!
