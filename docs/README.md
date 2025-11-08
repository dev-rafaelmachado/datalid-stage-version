## 📖 Índice da Documentação

### �🚀 Começando
- **[Guia de Início Rápido](01-QUICK-START.md)** - Comece em 5 minutos
- **[Instalação e Configuração](02-INSTALLATION.md)** - Setup completo do ambiente
- **[Primeiros Passos](03-FIRST-STEPS.md)** - Seus primeiros testes

### 🎯 Fundamentos
- **[Arquitetura do Sistema](04-ARCHITECTURE.md)** - Como tudo funciona
- **[Teoria e Conceitos](05-THEORY.md)** - Base teórica (YOLO, OCR, Deep Learning)
- **[Fluxo de Dados](06-DATA-FLOW.md)** - Jornada dos dados pelo sistema

### 🔧 Componentes Principais
- **[Detecção YOLO](07-YOLO-DETECTION.md)** - Detecção e segmentação de regiões
- **[Sistema OCR](08-OCR-SYSTEM.md)** - Extração de texto (7 engines disponíveis)
- **[Pré-processamento](09-PREPROCESSING.md)** - Preparação de imagens
- **[Pós-processamento](10-POST-PROCESSING.md)** - Validação e parsing de datas
- **[Pipeline Completo](11-FULL-PIPELINE.md)** - Integração end-to-end

### 📊 Treinamento e Avaliação
- **[Preparação de Dados](12-DATA-PREPARATION.md)** - Dataset e anotações
- **[Treinamento YOLO](13-YOLO-TRAINING.md)** - Treinar modelos customizados
- **[Avaliação de Performance](14-EVALUATION.md)** - Métricas e análise
- **[Comparação de Modelos](15-MODEL-COMPARISON.md)** - Escolher o melhor modelo

### 🌐 API e Integrações
- **[API REST](16-API-REST.md)** - Endpoints e uso da API
- **[Cliente Python](17-PYTHON-CLIENT.md)** - SDK Python
- **[Integrações](18-INTEGRATIONS.md)** - JavaScript, cURL, outras linguagens

### 🔬 Avançado
- **[Configurações YAML](19-YAML-CONFIG.md)** ✅ ✨ - Customização profunda
- **[OCR Engines Deep Dive](20-OCR-ENGINES.md)** ✅ ✨ - Análise detalhada de todos os engines OCR
- **[Otimização de Performance](21-OPTIMIZATION.md)** ✅ ✨ **NEW** - GPU, batching, caching, profiling
- **[Troubleshooting](22-TROUBLESHOOTING.md)** ✅ - Resolução de problemas


## 🎯 Navegação Rápida

### Por Objetivo

**Quero testar rapidamente:**
→ [Guia de Início Rápido](01-QUICK-START.md)

**Quero entender como funciona:**
→ [Arquitetura](04-ARCHITECTURE.md) + [Teoria](05-THEORY.md)

**Quero treinar meu próprio modelo:**
→ [Preparação de Dados](12-DATA-PREPARATION.md) + [Treinamento YOLO](13-YOLO-TRAINING.md)

**Quero integrar em minha aplicação:**
→ [API REST](16-API-REST.md) + [Cliente Python](17-PYTHON-CLIENT.md)

**Quero melhorar a precisão:**
→ [Otimização](21-OPTIMIZATION.md) + [OCR Engines](20-OCR-ENGINES.md)

### Por Nível de Experiência

**🌱 Iniciante:**
1. [Guia de Início Rápido](01-QUICK-START.md)
2. [Instalação](02-INSTALLATION.md)
3. [Primeiros Passos](03-FIRST-STEPS.md)
4. [Exemplos Práticos](23-EXAMPLES.md)

**🌿 Intermediário:**
1. [Arquitetura](04-ARCHITECTURE.md)
2. [Pipeline Completo](11-FULL-PIPELINE.md)
3. [Configurações YAML](19-YAML-CONFIG.md)
4. [API REST](16-API-REST.md)

**🌳 Avançado:**
1. [Teoria e Conceitos](05-THEORY.md)
2. [Treinamento YOLO](13-YOLO-TRAINING.md)
3. [Otimização](21-OPTIMIZATION.md)

## 📊 Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    DATALID 3.0 PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Imagem de      │
                    │  Entrada        │
                    └────────┬─────────┘
                             │
                             ▼
           ┌─────────────────────────────────┐
           │   ETAPA 1: DETECÇÃO YOLO       │
           │   • Localiza região da data    │
           │   • Segmentação poligonal      │
           │   • Extrai crop da região      │
           └────────────┬────────────────────┘
                        │
                        ▼
           ┌─────────────────────────────────┐
           │   ETAPA 2: PRÉ-PROCESSAMENTO   │
           │   • Normalização               │
           │   • Deskew / Perspectiva       │
           │   • Binarização / CLAHE        │
           │   • Remoção de ruído           │
           └────────────┬────────────────────┘
                        │
                        ▼
           ┌─────────────────────────────────┐
           │   ETAPA 3: OCR (7 Engines)     │
           │   • OpenOCR (recomendado)      │
           │   • PARSeq Enhanced            │
           │   • TrOCR                      │
           │   • EasyOCR / PaddleOCR        │
           │   • Tesseract                  │
           └────────────┬────────────────────┘
                        │
                        ▼
           ┌─────────────────────────────────┐
           │   ETAPA 4: PÓS-PROCESSAMENTO   │
           │   • Parsing de datas           │
           │   • Validação de formatos      │
           │   • Fuzzy matching             │
           │   • Score de confiança         │
           └────────────┬────────────────────┘
                        │
                        ▼
                    ┌──────────────────┐
                    │  Data Extraída  │
                    │  + Confiança    │
                    └──────────────────┘
```

### Recém Criados ✨

- ✅ [Instalação e Configuração](02-INSTALLATION.md) - Setup completo
- ✅ [Primeiros Passos](03-FIRST-STEPS.md) - Seus primeiros testes
- ✅ [Fluxo de Dados](06-DATA-FLOW.md) - Jornada dos dados
- ✅ [Detecção YOLO](07-YOLO-DETECTION.md) - Detecção e segmentação
- ✅ [Pré-processamento](09-PREPROCESSING.md) - Preparação de imagens
- ✅ [Pós-processamento](10-POST-PROCESSING.md) - Validação e parsing
- ✅ [Pipeline Completo](11-FULL-PIPELINE.md) - Integração end-to-end
- ✅ [Preparação de Dados](12-DATA-PREPARATION.md) - Dataset e anotações

## 🎓 Conceitos Fundamentais

### 1. **Detecção vs Segmentação**
- **Detecção**: BBox retangular ao redor da região
- **Segmentação**: Máscara poligonal precisa (usado neste projeto)

### 2. **Pipeline Modular**
Cada componente pode ser substituído/configurado independentemente:
- Trocar modelo YOLO (nano/small/medium/large)
- Trocar engine OCR (7 opções disponíveis)
- Customizar pré/pós-processamento

### 3. **Configuração por YAML**
Tudo é configurável via arquivos YAML em `config/`:
```yaml
detection:
  model_path: path/to/model.pt
  confidence: 0.25
ocr:
  engine: openocr
  preprocessing: config/preprocessing/ppro-openocr.yaml
```

## 🔥 Features Principais

✅ **7 Engines OCR** - OpenOCR, PARSeq, TrOCR, EasyOCR, PaddleOCR, Tesseract, TrOCR  
✅ **Segmentação Poligonal** - Precisão máxima com YOLOv8-seg  
✅ **Pré-processamento Inteligente** - 15+ técnicas de preparação de imagem  
✅ **API REST Completa** - FastAPI com Swagger/ReDoc  
✅ **Batch Processing** - Processa múltiplas imagens  
✅ **Sistema de Avaliação** - Métricas end-to-end detalhadas  
✅ **Visualizações** - Resultados anotados e análises visuais  
✅ **Docker Support** - Deploy facilitado  
✅ **Makefile Completo** - 50+ comandos prontos  

