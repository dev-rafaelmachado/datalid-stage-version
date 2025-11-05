# 📋 Formatos de Ground Truth Suportados

## Visão Geral

O sistema de avaliação da pipeline suporta dois formatos de ground truth JSON:

---

## Formato 1: Simples (Antigo)

**Uso:** Para datasets onde cada imagem tem apenas uma anotação de texto.

```json
{
  "annotations": {
    "image1.jpg": "15/01/2024",
    "image2.jpg": "20/03/2025",
    "image3.jpg": "10/12/2023"
  }
}
```

### Estrutura
- **Chave raiz:** `annotations` (configurável via `json_key` no config)
- **Chave:** Nome da imagem
- **Valor:** Texto esperado (data de validade)

---

## Formato 2: Detalhado (Novo)

**Uso:** Para datasets com múltiplas detecções por imagem e metadados adicionais.

```json
{
  "train_101_jpg.rf.abc123_box0": {
    "image": "train_101.jpg.rf.abc123.jpg",
    "expiry_date": "2024-01-15",
    "box_index": 0,
    "confidence": 0.95,
    "other_metadata": "..."
  },
  "train_102_jpg.rf.def456_box0": {
    "image": "train_102.jpg.rf.def456.jpg",
    "expiry_date": "2025-03-20",
    "box_index": 0,
    "confidence": 0.92
  }
}
```

### Estrutura
- **Chave raiz:** ID da detecção (qualquer string única)
- **Campos obrigatórios:**
  - `image`: Nome do arquivo da imagem
  - `expiry_date`: Data de validade esperada
- **Campos opcionais:**
  - `box_index`: Índice da caixa de detecção
  - `confidence`: Confiança da anotação
  - Outros metadados personalizados

### Múltiplas Detecções por Imagem

Quando uma imagem tem múltiplas detecções, o sistema **seleciona a primeira** encontrada:

```json
{
  "train_101_jpg.rf.abc123_box0": {
    "image": "train_101.jpg.rf.abc123.jpg",
    "expiry_date": "2024-01-15"
  },
  "train_101_jpg.rf.abc123_box1": {
    "image": "train_101.jpg.rf.abc123.jpg",
    "expiry_date": "2024-02-20"
  }
}
```

> ⚠️ **Nota:** Apenas `train_101_jpg.rf.abc123_box0` será usado para avaliação.

---

## Configuração

No arquivo `config/pipeline/pipeline_evaluation.yaml`:

```yaml
validation:
  ground_truth_format:
    type: json
    encoding: utf-8
    
    # Para Formato 1 (antigo)
    json_key: annotations  # Key que contém as anotações
    
    # Para Formato 2 (novo)
    json_key: annotations  # Opcional, pode ser omitido
    field_name: expiry_date  # Nome do campo com a data
    
    fields:
      image_id: filename
      label: text
      expiry: expiry_date
```

### Parâmetros

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `json_key` | Chave raiz para Formato 1 | `annotations` |
| `field_name` | Campo com a data no Formato 2 | `expiry_date` |

---

## Detecção Automática de Formato

O sistema detecta automaticamente qual formato está sendo usado:

1. **Verifica `json_key`:** Se existe e contém um dict simples → Formato 1
2. **Verifica estrutura:** Se cada valor tem campo `image` → Formato 2
3. **Fallback:** Assume Formato 1 direto (sem `annotations`)

### Exemplos de Detecção

```python
# Formato 1 detectado
{
  "annotations": {
    "img.jpg": "text"
  }
}

# Formato 2 detectado
{
  "det_1": {
    "image": "img.jpg",
    "expiry_date": "..."
  }
}

# Formato 1 direto detectado
{
  "img.jpg": "text"
}
```

---

## Validação de Formato

O script valida automaticamente:

✅ **Formato 1:**
- Chave `annotations` existe (se configurado)
- Valores são strings

✅ **Formato 2:**
- Cada entrada tem campo `image`
- Campo `expiry_date` existe (ou campo configurado)

❌ **Erros comuns:**
- Arquivo JSON inválido
- Estrutura não reconhecida
- Campos obrigatórios ausentes

---

## Exemplos Práticos

### Criar Ground Truth - Formato 1

```json
{
  "annotations": {
    "sample1.jpg": "15/01/2024",
    "sample2.jpg": "20/03/2025"
  }
}
```

### Criar Ground Truth - Formato 2

```json
{
  "detection_001": {
    "image": "sample1.jpg",
    "expiry_date": "2024-01-15"
  },
  "detection_002": {
    "image": "sample2.jpg",
    "expiry_date": "2025-03-20"
  }
}
```

### Usar no Pipeline

```bash
# Configurar path no YAML
dataset:
  ground_truth: data/ocr_test/ground_truth.json

# Executar avaliação
make evaluate-pipeline-quick
```

---

## Conversão entre Formatos

### Formato 1 → Formato 2

```python
import json

# Ler Formato 1
with open('gt_format1.json') as f:
    data = json.load(f)

# Converter para Formato 2
format2 = {}
for i, (image, date) in enumerate(data['annotations'].items()):
    format2[f"detection_{i:03d}"] = {
        "image": image,
        "expiry_date": date
    }

# Salvar
with open('gt_format2.json', 'w') as f:
    json.dump(format2, f, indent=2)
```

### Formato 2 → Formato 1

```python
import json

# Ler Formato 2
with open('gt_format2.json') as f:
    data = json.load(f)

# Converter para Formato 1 (pega primeira detecção por imagem)
annotations = {}
for det_id, det_data in data.items():
    image = det_data['image']
    if image not in annotations:
        annotations[image] = det_data['expiry_date']

format1 = {"annotations": annotations}

# Salvar
with open('gt_format1.json', 'w') as f:
    json.dump(format1, f, indent=2)
```

---

## Referências

- **Configuração:** `config/pipeline/pipeline_evaluation.yaml`
- **Script:** `scripts/pipeline/evaluate_pipeline.py`
- **Função:** `load_ground_truth()` na classe `PipelineEvaluator`