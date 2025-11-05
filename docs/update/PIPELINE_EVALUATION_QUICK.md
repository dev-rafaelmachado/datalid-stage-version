# ⚡ Quick Start: Pipeline Evaluation

Guia rápido para avaliar a pipeline completa (YOLO + OCR + Parsing).

---

## 📋 Pré-requisitos

1. **Ground Truth JSON** preparado
2. **Imagens de teste** no diretório configurado
3. **Pipeline funcionando** (testada previamente)

---

## 🚀 Execução Rápida

### Avaliação Completa (Todas as Imagens)

```bash
make evaluate-pipeline-quick
```

### Avaliação com N Imagens

```bash
make evaluate-pipeline-quick NUM_IMAGES=10
```

### Avaliação Customizada

```bash
make evaluate-pipeline-custom \
  GT_FILE=data/custom/ground_truth.json \
  IMAGES_DIR=data/custom/images \
  NUM_IMAGES=50
```

---

## 📁 Estrutura de Dados Necessária

### Ground Truth JSON

**Formato 1 (Simples):**
```json
{
  "annotations": {
    "image1.jpg": "15/01/2024",
    "image2.jpg": "20/03/2025"
  }
}
```

**Formato 2 (Detalhado - Novo ✨):**
```json
{
  "train_101_jpg.rf.abc123_box0": {
    "image": "train_101.jpg",
    "expiry_date": "2024-01-15"
  }
}
```

> 📖 Veja [GROUND_TRUTH_FORMAT.md](GROUND_TRUTH_FORMAT.md) para detalhes completos.

### Estrutura de Diretórios

```
data/ocr_test/
├── ground_truth.json          # Anotações
└── images/                    # Imagens de teste
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## 📊 Outputs Gerados

Após a execução, você encontrará:

```
outputs/pipeline_evaluation/YYYY-MM-DD_HH-MM-SS/
├── results/
│   ├── detailed_results.csv       # Resultados por imagem
│   ├── metrics_summary.json       # Métricas agregadas
│   └── report.md                  # Relatório legível
├── visualizations/
│   ├── metrics_overview.png
│   ├── confusion_matrix.png
│   └── error_distribution.png
├── intermediate_steps/
│   └── [imagem]/
│       ├── 1_input.jpg
│       ├── 2_detection.jpg
│       ├── 3_crops.jpg
│       └── ...
└── errors/
    ├── error_analysis.json
    └── error_examples/
```

---

## 📈 Métricas Principais

| Métrica | Descrição |
|---------|-----------|
| **Pipeline Accuracy** | % de datas corretas no final |
| **Detection Rate** | % de imagens com detecção YOLO |
| **OCR Exact Match** | % de textos idênticos ao GT |
| **Date Found Rate** | % de datas encontradas no parsing |
| **Processing Time** | Tempo médio por imagem |

---

## 🔧 Configuração Rápida

Edite `config/pipeline/pipeline_evaluation.yaml`:

```yaml
dataset:
  images_dir: data/ocr_test/images
  ground_truth: data/ocr_test/ground_truth.json
  num_images: null  # null = todas, ou número específico
  selection_mode: all  # all, first, random

validation:
  ground_truth_format:
    json_key: annotations  # Para Formato 1
    field_name: expiry_date  # Para Formato 2
```

---

## 🐛 Troubleshooting

### Erro: Ground Truth não encontrado

```bash
# Verificar path
ls data/ocr_test/ground_truth.json

# Ajustar no config
dataset:
  ground_truth: <path_correto>
```

### Erro: Formato JSON inválido

```bash
# Validar JSON
python -m json.tool data/ocr_test/ground_truth.json
```

### Erro: Pipeline não inicializada

```bash
# Verificar config da pipeline
cat config/pipeline/full_pipeline.yaml

# Testar pipeline isoladamente
python scripts/pipeline/run_full_pipeline.py --image data/sample.jpg
```

### Nenhuma imagem processada

```bash
# Verificar imagens no diretório
ls data/ocr_test/images/

# Verificar extensões suportadas (.jpg, .jpeg, .png)
```

---

## 📚 Referências

- **Documentação Completa:** [PIPELINE_EVALUATION.md](PIPELINE_EVALUATION.md)
- **Formatos Ground Truth:** [GROUND_TRUTH_FORMAT.md](GROUND_TRUTH_FORMAT.md)
- **Checklist de Validação:** [PIPELINE_EVALUATION_CHECKLIST.md](PIPELINE_EVALUATION_CHECKLIST.md)
- **Implementação Técnica:** [PIPELINE_EVALUATION_IMPLEMENTATION.md](PIPELINE_EVALUATION_IMPLEMENTATION.md)

---

## 🎯 Próximos Passos

1. ✅ Execute a avaliação rápida
2. 📊 Analise as métricas geradas
3. 🔍 Investigue erros no relatório
4. ⚙️ Ajuste configs baseado nos resultados
5. 🔄 Re-execute e compare melhorias

---

> 💡 **Dica:** Comece com `NUM_IMAGES=5` para validar o setup antes de rodar todas as imagens!
