# 📊 Avaliação de Performance

> Métricas, análise e benchmarking de modelos

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Métricas de Detecção](#métricas-de-detecção)
- [Métricas de OCR](#métricas-de-ocr)
- [Métricas End-to-End](#métricas-end-to-end)
- [Ferramentas de Avaliação](#ferramentas-de-avaliação)
- [Análise de Erros](#análise-de-erros)
- [Benchmarking](#benchmarking)

## 🎯 Visão Geral

Avaliar a performance do sistema em três níveis:
1. **Detecção (YOLO)** - Quão bem localiza regiões?
2. **OCR** - Quão bem extrai texto?
3. **End-to-End** - Quão bem extrai datas completas?

## 📍 Métricas de Detecção

### IoU (Intersection over Union)

**Definição:** Sobreposição entre predição e ground truth

```
IoU = Área de Interseção / Área de União

       ┌───────────┐
       │ Pred      │
       │  ┌────────┼────┐
       │  │████████│    │ GT
       └──┼────────┘    │
          └─────────────┘
          
IoU = Área Roxa / (Área Pred + Área GT - Área Roxa)
```

**Código:**
```python
def calculate_iou(box1, box2):
    """
    Calcula IoU entre duas boxes.
    box: [x1, y1, x2, y2]
    """
    # Intersecção
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # União
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

**Interpretação:**
- IoU > 0.5: Boa detecção
- IoU > 0.7: Ótima detecção
- IoU > 0.9: Perfeita

### mAP (mean Average Precision)

**Definição:** Métrica principal para detecção de objetos

```python
# mAP@0.5: Média de AP com IoU threshold 0.5
# mAP@0.5:0.95: Média de AP de IoU 0.5 a 0.95 (passo 0.05)

def calculate_map(detections, ground_truths, iou_threshold=0.5):
    """Calcula mAP."""
    # 1. Para cada classe
    # 2. Ordenar detecções por confiança
    # 3. Calcular TP, FP para cada threshold
    # 4. Calcular Precision e Recall
    # 5. Calcular AP (área sob curva PR)
    # 6. Média de AP de todas as classes
    pass
```

**Via Ultralytics:**
```python
from ultralytics import YOLO

model = YOLO('experiments/best.pt')
metrics = model.val(data='data.yaml')

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"mAP@0.75: {metrics.box.map75:.4f}")
```

### Precision e Recall

```
Precision = TP / (TP + FP)  # Precisão
Recall = TP / (TP + FN)     # Revocação

TP (True Positive): Detecções corretas
FP (False Positive): Detecções incorretas
FN (False Negative): Ground truths não detectados
```

**Trade-off:**
- Alta Precision: Poucas detecções falsas
- High Recall: Poucas detecções perdidas
- F1-Score: Média harmônica de ambos

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

### Confusion Matrix

```
                Predicted
              Pos      Neg
Actual  Pos   TP       FN
        Neg   FP       TN
```

**Gerar:**
```python
model = YOLO('best.pt')
metrics = model.val(plots=True)
# Salva em: runs/val/confusion_matrix.png
```

## 📝 Métricas de OCR

### CER (Character Error Rate)

**Definição:** Taxa de erro por caractere

```python
from difflib import SequenceMatcher

def calculate_cer(predicted, ground_truth):
    """
    Calcula Character Error Rate.
    CER = (S + D + I) / N
    S: Substituições
    D: Deleções
    I: Inserções
    N: Total de caracteres no ground truth
    """
    # Levenshtein distance
    m, n = len(predicted), len(ground_truth)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i-1] == ground_truth[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Deleção
                    dp[i][j-1],    # Inserção
                    dp[i-1][j-1]   # Substituição
                )
    
    edit_distance = dp[m][n]
    cer = edit_distance / len(ground_truth)
    return cer

# Exemplo
pred = "15i03i2025"
gt = "15/03/2025"
cer = calculate_cer(pred, gt)
print(f"CER: {cer:.2%}")  # 20% (2 erros em 10 chars)
```

**Interpretação:**
- CER < 5%: Excelente
- CER < 10%: Bom
- CER < 20%: Aceitável
- CER > 20%: Ruim

### WER (Word Error Rate)

**Definição:** Taxa de erro por palavra

```python
def calculate_wer(predicted, ground_truth):
    """
    Calcula Word Error Rate.
    Similar ao CER, mas por palavras.
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    # Mesmo algoritmo do CER, mas para palavras
    edit_distance = levenshtein_distance(pred_words, gt_words)
    wer = edit_distance / len(gt_words)
    return wer

# Exemplo
pred = "VAL 15i03i2025"
gt = "VAL 15/03/2025"
wer = calculate_wer(pred, gt)
print(f"WER: {wer:.2%}")  # 50% (1 erro em 2 palavras)
```

### Accuracy

```python
def calculate_accuracy(predictions, ground_truths):
    """
    Acurácia: % de predições completamente corretas.
    """
    correct = sum(p == gt for p, gt in zip(predictions, ground_truths))
    accuracy = correct / len(ground_truths)
    return accuracy
```

### Confidence Score

```python
def analyze_confidence(predictions):
    """Analisa distribuição de confiança."""
    confidences = [p['confidence'] for p in predictions]
    
    stats = {
        'mean': np.mean(confidences),
        'median': np.median(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences)
    }
    
    return stats
```

## 🎯 Métricas End-to-End

### Date Extraction Accuracy

```python
def evaluate_date_extraction(results, ground_truths):
    """
    Avalia extração de data completa.
    """
    metrics = {
        'total': len(ground_truths),
        'correct': 0,
        'wrong_date': 0,
        'no_detection': 0,
        'ocr_failed': 0
    }
    
    for result, gt in zip(results, ground_truths):
        if not result['detections']:
            metrics['no_detection'] += 1
        elif not result['best_date']:
            metrics['ocr_failed'] += 1
        elif result['best_date']['date'] == gt['date']:
            metrics['correct'] += 1
        else:
            metrics['wrong_date'] += 1
    
    metrics['accuracy'] = metrics['correct'] / metrics['total']
    
    return metrics
```

### Pipeline Performance

```python
{
    'detection_rate': 0.95,      # 95% das imagens tiveram detecção
    'ocr_success_rate': 0.92,    # 92% das detecções geraram texto
    'parse_success_rate': 0.88,  # 88% dos textos foram parseados
    'end_to_end_accuracy': 0.85, # 85% de datas corretas
    'avg_processing_time': 1.2,  # 1.2s por imagem
    'avg_confidence': 0.89       # Confiança média 89%
}
```

## 🛠️ Ferramentas de Avaliação

### Script de Avaliação YOLO

```python
# scripts/evaluation/evaluate_yolo.py
from ultralytics import YOLO
from pathlib import Path
import json

def evaluate_yolo_model(model_path, data_yaml):
    """Avalia modelo YOLO."""
    model = YOLO(model_path)
    
    # Validar
    metrics = model.val(
        data=data_yaml,
        split='test',
        plots=True,
        save_json=True
    )
    
    # Extrair métricas
    results = {
        'model': model_path,
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.p),
        'recall': float(metrics.box.r),
        'f1': float(2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r))
    }
    
    print(json.dumps(results, indent=2))
    return results

# Usar
evaluate_yolo_model(
    'experiments/yolov8n_seg_v1/weights/best.pt',
    'data/dataset_v1/data.yaml'
)
```

### Script de Avaliação OCR

```python
# scripts/evaluation/evaluate_ocr.py
import json
from pathlib import Path
from src.ocr.engines.openocr import OpenOCREngine

def evaluate_ocr_engine(engine, test_images, ground_truths):
    """Avalia engine OCR."""
    results = []
    
    for img_path, gt in zip(test_images, ground_truths):
        # OCR
        pred_text, conf = engine.recognize(img_path)
        
        # Métricas
        cer = calculate_cer(pred_text, gt['text'])
        exact_match = pred_text == gt['text']
        
        results.append({
            'image': str(img_path),
            'predicted': pred_text,
            'ground_truth': gt['text'],
            'confidence': conf,
            'cer': cer,
            'exact_match': exact_match
        })
    
    # Resumo
    summary = {
        'total': len(results),
        'exact_matches': sum(r['exact_match'] for r in results),
        'accuracy': sum(r['exact_match'] for r in results) / len(results),
        'avg_cer': sum(r['cer'] for r in results) / len(results),
        'avg_confidence': sum(r['confidence'] for r in results) / len(results)
    }
    
    return results, summary
```

### Script de Avaliação Pipeline Completo

```python
# scripts/evaluation/evaluate_pipeline.py
from src.pipeline.full_pipeline import FullPipeline
from src.ocr.config import load_pipeline_config
import json
from pathlib import Path

def evaluate_full_pipeline(config_path, test_data_path):
    """
    Avalia pipeline completo.
    
    test_data_path: Caminho para data/evaluation/ground_truth.json
    """
    # Carregar pipeline
    config = load_pipeline_config(config_path)
    pipeline = FullPipeline(config)
    
    # Carregar ground truth
    with open(test_data_path) as f:
        ground_truths = json.load(f)
    
    results = []
    for gt in ground_truths:
        img_path = gt['image_path']
        
        # Processar
        result = pipeline.process(img_path)
        
        # Comparar
        correct = False
        if result['best_date']:
            correct = result['best_date']['date'] == gt['date']
        
        results.append({
            'image': img_path,
            'predicted': result['best_date']['date'] if result['best_date'] else None,
            'ground_truth': gt['date'],
            'correct': correct,
            'confidence': result['best_date']['confidence'] if result['best_date'] else 0,
            'processing_time': result['processing_time']['total']
        })
    
    # Métricas
    metrics = {
        'total_images': len(results),
        'correct_predictions': sum(r['correct'] for r in results),
        'accuracy': sum(r['correct'] for r in results) / len(results),
        'avg_confidence': sum(r['confidence'] for r in results) / len(results),
        'avg_time': sum(r['processing_time'] for r in results) / len(results),
        'detection_rate': sum(1 for r in results if r['predicted']) / len(results)
    }
    
    # Salvar
    output = {
        'metrics': metrics,
        'results': results
    }
    
    output_path = 'outputs/evaluation/pipeline_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(json.dumps(metrics, indent=2))
    return metrics, results

# Usar
evaluate_full_pipeline(
    'config/pipeline/full_pipeline.yaml',
    'data/evaluation/ground_truth.json'
)
```

### Via Makefile

```bash
# Avaliar YOLO
make yolo-evaluate MODEL=experiments/best.pt

# Avaliar OCR
make ocr-evaluate ENGINE=openocr

# Avaliar pipeline completo
make pipeline-evaluate CONFIG=config/pipeline/full_pipeline.yaml
```

## 🔍 Análise de Erros

### Categorização de Erros

```python
def categorize_errors(results, ground_truths):
    """Categoriza tipos de erros."""
    errors = {
        'no_detection': [],      # YOLO não detectou
        'wrong_location': [],    # Detecção incorreta
        'ocr_error': [],         # OCR falhou
        'parse_error': [],       # Parse falhou
        'format_error': []       # Formato incorreto
    }
    
    for result, gt in zip(results, ground_truths):
        if not result['detections']:
            errors['no_detection'].append({
                'image': result['image_path'],
                'reason': 'No detections by YOLO'
            })
        elif not result['ocr_results']:
            errors['ocr_error'].append({
                'image': result['image_path'],
                'reason': 'OCR failed to extract text'
            })
        elif not result['parsed_dates']:
            errors['parse_error'].append({
                'image': result['image_path'],
                'text': result['ocr_results'][0]['text'],
                'reason': 'Failed to parse date from text'
            })
        elif result['best_date']['date'] != gt['date']:
            errors['format_error'].append({
                'image': result['image_path'],
                'predicted': result['best_date']['date'],
                'expected': gt['date']
            })
    
    return errors
```

### Visualização de Erros

```python
# scripts/evaluation/visualize_errors.py
import cv2
import matplotlib.pyplot as plt

def visualize_errors(errors, output_dir):
    """Cria visualizações de erros."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Para cada tipo de erro
    for error_type, error_list in errors.items():
        if not error_list:
            continue
        
        # Criar grid de imagens com erros
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{error_type} ({len(error_list)} errors)')
        
        for idx, error in enumerate(error_list[:6]):  # Primeiros 6
            ax = axes[idx // 3, idx % 3]
            
            # Carregar imagem
            img = cv2.imread(error['image'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Exibir
            ax.imshow(img)
            ax.set_title(error.get('reason', 'Error'))
            ax.axis('off')
        
        # Salvar
        plt.tight_layout()
        plt.savefig(output_dir / f'{error_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
```

## 📈 Benchmarking

### Comparar Modelos YOLO

```python
# scripts/evaluation/benchmark_yolo.py
models = [
    'yolov8n-seg.pt',
    'yolov8s-seg.pt',
    'yolov8m-seg.pt'
]

results = []
for model_path in models:
    metrics = evaluate_yolo_model(model_path, 'data/dataset_v1/data.yaml')
    results.append(metrics)

# Comparar
import pandas as pd
df = pd.DataFrame(results)
print(df.to_markdown())
```

### Comparar OCR Engines

```python
# scripts/evaluation/benchmark_ocr_engines.py
from src.ocr.engines import get_all_engines

engines = get_all_engines()
results = []

for engine_name, engine in engines.items():
    summary = evaluate_ocr_engine(engine, test_images, ground_truths)
    summary['engine'] = engine_name
    results.append(summary)

# Comparar
df = pd.DataFrame(results)
df = df.sort_values('accuracy', ascending=False)
print(df)
```

### Comparar Configurações de Pipeline

```bash
# Benchmark diferentes configs
for config in config/pipeline/*.yaml; do
    echo "Testing $config..."
    make pipeline-evaluate CONFIG=$config
done
```

## 📊 Relatório de Avaliação

### Gerar Relatório Completo

```python
# scripts/evaluation/generate_report.py
def generate_evaluation_report(results, output_path):
    """Gera relatório HTML de avaliação."""
    html = f"""
    <html>
    <head><title>Datalid Evaluation Report</title></head>
    <body>
        <h1>Evaluation Report</h1>
        
        <h2>Overall Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>{results['accuracy']:.2%}</td></tr>
            <tr><td>mAP@0.5</td><td>{results['map50']:.4f}</td></tr>
            <tr><td>CER</td><td>{results['cer']:.2%}</td></tr>
        </table>
        
        <h2>Error Analysis</h2>
        <img src="error_distribution.png">
        
        <h2>Sample Results</h2>
        ...
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
```

## 📚 Referências

- [YOLO Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [OCR Evaluation](https://github.com/clovaai/deep-text-recognition-benchmark)

## 💡 Próximos Passos

- **[Comparação de Modelos](15-MODEL-COMPARISON.md)** - Escolher melhor modelo
- **[Otimização](21-OPTIMIZATION.md)** - Melhorar performance
- **[Troubleshooting](22-TROUBLESHOOTING.md)** - Resolver problemas

---

**Dúvidas sobre avaliação?** Consulte [FAQ](25-FAQ.md)
