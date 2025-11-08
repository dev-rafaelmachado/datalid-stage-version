# 🎯 Datalid 3.0

> Sistema Inteligente de Detecção e Extração de Datas de Validade usando Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
[![Documentation](https://img.shields.io/badge/docs-complete-brightgreen.svg)](docs/README.md)

Sistema **modular**, **escalável** e de **alto desempenho** que combina **YOLOv8** (detecção/segmentação) com **7 engines OCR** e pós-processamento inteligente para extrair datas de validade de produtos com **45%+ de precisão**.

---

## ✨ Destaques

✅ **Segmentação Poligonal Precisa** - YOLOv8-seg para máxima acurácia  
✅ **7 Engines OCR** - OpenOCR, PARSeq, TrOCR, EasyOCR, PaddleOCR, Tesseract  
✅ **Pipeline End-to-End** - Detecção → Pré-processamento → OCR → Validação  
✅ **API REST Completa** - FastAPI com Swagger/ReDoc  
✅ **Configurável via YAML** - Customize sem tocar no código  
✅ **Sistema de Avaliação** - Métricas detalhadas (CER, WER, IoU, F1)  
✅ **Produção-Ready** - Docker, rate limiting, monitoramento  

---

## 🚀 Início Rápido (5 minutos)

### Instalação

```bash
# Clone o repositório
git clone [seu-repo]
cd datalid3.0

# Instale as dependências
pip install -r requirements.txt

# Valide o ambiente
make validate-env
```

### Primeiro Teste

```bash
# Teste em uma imagem de exemplo
make pipeline-test IMAGE=data/ocr_test/sample.jpg
```

**Resultado esperado:**
```
✅ Data encontrada: 15/03/2025
   Confiança: 95%
   Tempo: 1.2s
```

### Teste com sua imagem

```bash
make pipeline-test IMAGE=/caminho/para/sua/imagem.jpg
```

---

## 📖 Documentação Completa

A documentação está **completamente atualizada** e organizada em **[docs/README.md](docs/README.md)** com 25+ guias detalhados.

### 📚 Guias Principais

| Guia | Descrição | Nível |
|------|-----------|-------|
| **[Início Rápido](docs/01-QUICK-START.md)** | Comece em 5 minutos | 🌱 Iniciante |
| **[Instalação](docs/02-INSTALLATION.md)** | Setup completo do ambiente | 🌱 Iniciante |
| **[Primeiros Passos](docs/03-FIRST-STEPS.md)** | Seus primeiros testes | 🌱 Iniciante |
| **[Arquitetura](docs/04-ARCHITECTURE.md)** | Como o sistema funciona | 🌿 Intermediário |
| **[Teoria](docs/05-THEORY.md)** | YOLO, OCR e Deep Learning | 🌳 Avançado |
| **[Pipeline Completo](docs/11-FULL-PIPELINE.md)** | Integração end-to-end | 🌿 Intermediário |
| **[Preparação de Dados](docs/12-DATA-PREPARATION.md)** | Dataset e anotações | 🌳 Avançado |

### 🎯 Navegação por Objetivo

**🧪 Quero testar rapidamente:**  
→ [Guia de Início Rápido](docs/01-QUICK-START.md) → [Primeiros Passos](docs/03-FIRST-STEPS.md)

**🎓 Quero entender como funciona:**  
→ [Arquitetura](docs/04-ARCHITECTURE.md) → [Teoria](docs/05-THEORY.md) → [Fluxo de Dados](docs/06-DATA-FLOW.md)

**🔬 Quero treinar meu próprio modelo:**  
→ [Preparação de Dados](docs/12-DATA-PREPARATION.md) → [Treinamento YOLO](docs/13-YOLO-TRAINING.md)

**🌐 Quero integrar em minha aplicação:**  
→ [API REST](docs/16-API-REST.md) → [Cliente Python](docs/17-PYTHON-CLIENT.md)

**⚡ Quero melhorar a precisão:**  
→ [OCR Engines](docs/20-OCR-ENGINES.md) → [Otimização](docs/21-OPTIMIZATION.md)  

---

## 🎯 Como Funciona

```
┌─────────────┐
│   Imagem    │  (Foto do produto)
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  1. DETECÇÃO YOLO      │  → Localiza região da data
│     (Segmentação)       │    (máscara poligonal)
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  2. PRÉ-PROCESSAMENTO  │  → Melhora qualidade
│     • Deskew            │    (contraste, rotação,
│     • CLAHE             │     binarização)
│     • Denoise           │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  3. OCR                │  → Extrai texto
│     (7 engines)         │    "VAL: 15/03/2025"
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  4. PÓS-PROCESSAMENTO  │  → Valida e parsea
│     • Regex             │    Data: 15/03/2025
│     • Fuzzy matching    │    Confiança: 45%
│     • Validação         │
└─────────────────────────┘
```

---

## 📊 Performance

### Precisão

| Componente | Métrica | Valor |
|------------|---------|-------|
| **Detecção YOLO** | mAP@0.5 | 93% |
| **OCR (OpenOCR)** | Acurácia | 72% |
| **End-to-End** | F1-Score | 45% |

### Velocidade (GPU RTX 3060)

| Modelo | FPS | Tempo/Imagem |
|--------|-----|--------------|
| YOLOv8n-seg | ~3.3 | 0.3s |
| YOLOv8s-seg | ~2.0 | 0.5s |
| YOLOv8m-seg | ~1.2 | 0.8s |

---

## 🔧 Estrutura do Projeto

```
datalid3.0/
├── 📁 src/                    # Código-fonte
│   ├── yolo/                  # Detecção e segmentação
│   ├── ocr/                   # 7 engines OCR
│   ├── pipeline/              # Pipelines end-to-end
│   ├── api/                   # API REST (FastAPI)
│   ├── data/                  # Processamento de dados
│   └── utils/                 # Utilitários
│
├── 📁 config/                 # Configurações YAML
│   ├── pipeline/              # Configs de pipeline
│   ├── yolo/                  # Configs YOLO
│   ├── ocr/                   # Configs OCR engines
│   └── preprocessing/         # Configs pré-processamento
│
├── 📁 scripts/                # Scripts utilitários
│   ├── pipeline/              # Scripts de pipeline
│   ├── training/              # Scripts de treino
│   ├── evaluation/            # Scripts de avaliação
│   ├── api/                   # Scripts de API
│   └── inference/             # Scripts de inferência
│
├── 📁 docs/                   # 📚 Documentação completa
│   ├── README.md              # Índice da documentação
│   ├── 01-QUICK-START.md      # Guia rápido
│   ├── 04-ARCHITECTURE.md     # Arquitetura do sistema
│   └── 05-THEORY.md           # Teoria e conceitos
│
├── 📁 data/                   # Dados
│   ├── raw/                   # Dados brutos
│   ├── processed/             # Datasets processados
│   └── evaluation/            # Ground truth
│
├── 📁 experiments/            # Experimentos e modelos
├── 📁 outputs/                # Resultados
├── Makefile                   # 50+ comandos prontos
└── requirements.txt           # Dependências
```

---

## ⚙️ Configuração e Customização

### Trocar Modelo YOLO

Edite `config/pipeline/full_pipeline.yaml`:

```yaml
detection:
  model_path: experiments/yolov8m_seg_best/weights/best.pt  # medium (padrão)
  # ou
  model_path: experiments/yolov8s_seg_best/weights/best.pt  # small (mais rápido)
  # ou  
  model_path: experiments/yolov8n_seg_best/weights/best.pt  # nano (muito rápido)
  confidence: 0.25
  iou: 0.7
```

### Trocar Engine OCR

```yaml
ocr:
  engine: openocr        # Padrão (71% precisão) ⭐
  # ou
  engine: parseq_enhanced  # PARSeq melhorado (30% precisão)
  # ou
  engine: trocr           # TrOCR (30% precisão)
  # ou
  engine: easyocr         # EasyOCR (14% precisão)
```

### Ajustar Pré-processamento

```yaml
ocr:
  preprocessing: config/preprocessing/ppro-openocr.yaml
  # ou
  preprocessing: config/preprocessing/ppro-minimal.yaml  # Mais rápido
```

---

## 📊 Avaliação da Pipeline

Sistema completo de avaliação end-to-end:

```bash
# Teste rápido (10 imagens)
make pipeline-eval-quick

# Avaliação customizada
make pipeline-eval NUM=20 MODE=random

# Avaliação completa
make pipeline-eval-full
```

**Métricas calculadas:**
- ✅ Detecção (YOLO): mAP, recall, precision
- ✅ OCR: CER, WER, exact match, similaridade
- ✅ Parsing: taxa de sucesso, formatos encontrados
- ✅ End-to-end: acurácia, F1-score, tempo

**Outputs gerados:**
- 📊 CSV com resultados detalhados
- 📈 Gráficos e visualizações
- 📝 Relatório markdown
- 🔍 Análise de erros por etapa

Veja [docs/14-EVALUATION.md](docs/14-EVALUATION.md) para detalhes.

---

## 🌐 API REST

Sistema completo de API REST para integração com aplicações.

### Início Rápido

```bash
# Iniciar API
make api-run

# API disponível em: http://localhost:8000
# Docs interativa: http://localhost:8000/docs
```

### Exemplo de Uso

**Python:**
```python
from scripts.api.client import DatalidClient

client = DatalidClient("http://localhost:8000")
result = client.process_image("produto.jpg")
print(f"Data: {result['best_date']['date']}")
print(f"Confiança: {result['best_date']['confidence']}")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@produto.jpg"
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/process', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Data:', data.best_date.date);
```

### Features da API

✅ **Swagger/ReDoc** - Documentação interativa  
✅ **7 Engines OCR** - Configurável por requisição  
✅ **Batch Processing** - Múltiplas imagens  
✅ **Rate Limiting** - Controle de uso  
✅ **Docker Ready** - Deploy facilitado  
✅ **Cliente Python** - SDK incluso  

### Endpoints Principais

```
POST   /process        # Processar imagem
POST   /batch          # Processar múltiplas
GET    /health         # Health check
GET    /models         # Listar modelos
GET    /engines        # Listar engines OCR
```

**Documentação completa:** [docs/16-API-REST.md](docs/16-API-REST.md)

---

## 🔧 Comandos Make Essenciais

```bash
# TESTES
make pipeline-test IMAGE=img.jpg    # Testar em uma imagem
make pipeline-eval-quick            # Avaliação rápida (10 imgs)
make pipeline-eval-full             # Avaliação completa

# OCR
make ocr-test ENGINE=openocr        # Testar engine específico
make ocr-compare                    # Comparar todos os engines

# API
make api-run                        # Iniciar API
make api-test                       # Testar API
make api-health                     # Verificar status

# TREINAMENTO
make train-small                    # Treinar YOLOv8s-seg
make train-medium                   # Treinar YOLOv8m-seg

# VALIDAÇÃO
make validate-env                   # Verificar ambiente
make test-cuda                      # Testar GPU

# VISUALIZAÇÃO
make tensorboard                    # Ver métricas
```

**Lista completa:** Execute `make help` ou veja o [Makefile](Makefile)

---

## 🎓 Conceitos Fundamentais

### Detecção vs Segmentação

**Detecção (BBox)**: Retângulo ao redor da região  
**Segmentação (Máscara)**: Contorno poligonal preciso ⭐ *Usamos este!*

Vantagem da segmentação: Remove fundo e ruído, melhorando OCR.

### Pipeline Modular

Cada componente é independente e configurável:
- **Modelo YOLO**: nano/small/medium/large
- **Engine OCR**: 7 opções disponíveis
- **Pré-processamento**: Customizável por engine
- **Pós-processamento**: Ajustável via regex/fuzzy

### Configuração por YAML

Tudo é configurável sem alterar código:

```yaml
detection:
  model_path: path/to/model.pt
  confidence: 0.25
  
ocr:
  engine: openocr
  preprocessing: config/preprocessing/ppro-openocr.yaml
  
parsing:
  min_confidence: 0.5
  fuzzy_threshold: 0.8
```

---

## 📚 Documentação

A documentação completa está em **[docs/README.md](docs/README.md)** e inclui:

### 🎯 Para Iniciantes
- [Guia de Início Rápido](docs/01-QUICK-START.md)
- [Instalação Completa](docs/02-INSTALLATION.md)
- [Primeiros Passos](docs/03-FIRST-STEPS.md)

### 🧠 Para Entender
- [Arquitetura do Sistema](docs/04-ARCHITECTURE.md)
- [Teoria e Conceitos](docs/05-THEORY.md)
- [Fluxo de Dados](docs/06-DATA-FLOW.md)

### 🔧 Para Usar
- [Sistema OCR](docs/08-OCR-SYSTEM.md)
- [Pipeline Completo](docs/11-FULL-PIPELINE.md)
- [Configurações YAML](docs/19-YAML-CONFIG.md)

### 🚀 Para Avançar
- [Treinamento YOLO](docs/13-YOLO-TRAINING.md)
- [Avaliação de Performance](docs/14-EVALUATION.md)
- [Otimização](docs/21-OPTIMIZATION.md)

---

## 🎯 Casos de Uso

### 1. Controle de Qualidade

```python
# Verificar datas de validade em lote
results = pipeline.process_directory("produtos/")

# Filtrar produtos próximos ao vencimento
proximos_vencer = [
    r for r in results 
    if r['best_date']['days_until_expiry'] < 30
]
```

### 2. Sistemas de Inventário

```python
# Integração com banco de dados
for produto in produtos:
    result = pipeline.process(produto.imagem)
    produto.data_validade = result['best_date']['date']
    produto.confianca = result['best_date']['confidence']
    produto.save()
```

### 3. Aplicativos Mobile

```javascript
// Upload via API
async function verificarValidade(foto) {
    const formData = new FormData();
    formData.append('file', foto);
    
    const response = await fetch('http://api.datalid.com/process', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    return data.best_date;
}
```
---

## 📝 Licença

Este projeto está sob licença MIT - veja [LICENSE](LICENSE) para detalhes.

## 🎓 Citação

Se você usar este projeto em sua pesquisa, por favor cite:

```bibtex
@software{datalid3.0,
  author = {Seu Nome},
  title = {Datalid 3.0: Sistema Inteligente de Detecção de Datas de Validade},
  year = {2025},
  url = {https://github.com/seu-usuario/datalid3.0}
}
```

---

<div align="center">

**[⬆ Voltar ao topo](#-datalid-30)**

</div>
