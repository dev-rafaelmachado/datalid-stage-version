# 🔍 Sistema OCR - Guia Completo

> Tudo sobre os 7 engines de OCR disponíveis no Datalid 3.0

## 📚 Índice

1. [Visão Geral](#visão-geral)
2. [Engines Disponíveis](#engines-disponíveis)
3. [Comparação Detalhada](#comparação-detalhada)
4. [Como Escolher](#como-escolher)
5. [Uso e Configuração](#uso-e-configuração)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

O Datalid 3.0 oferece **7 engines OCR** diferentes, cada um com suas características:

| Engine | Tipo | Velocidade | GPU | Recomendado |
|--------|------|----------|-----------|-----|-------------|
| **OpenOCR** | Deep Learning | Médio | ✅ Sim | ⭐ Produção |
| **PARSeq Enhanced** | Transformer | Médio | ✅ Sim | ⭐ Alternativa |
| **TrOCR** | Transformer | Lento | ✅ Sim | Pesquisa |
| **EasyOCR** | Deep Learning | Médio | ✅ Opcional | Uso geral |
| **PaddleOCR** | Deep Learning | Rápido | ✅ Opcional | Chinês |
| **Tesseract** | Tradicional | Muito rápido | ❌ Não | Testes |
| **PARSeq Base** | Transformer | Rápido | ✅ Sim | Base |

---

## 🚀 Engines Disponíveis

### 1. OpenOCR ⭐ (Recomendado)

**O melhor para produção**

#### Características
- Arquitetura: CNN + Transformer Encoder
- Treinado em: Scene text (textos em cena real)
- Idiomas: Multi-language (PT, EN, ES, FR, etc.)
- GPU: Recomendado

#### Por que é o melhor?
✅ Alta precisão em textos de embalagens  
✅ Robusto a perspectiva e deformações  
✅ Bom balanço precisão/velocidade  
✅ Suporta múltiplos idiomas nativamente  

#### Instalação
```bash
pip install openocr
```

#### Uso
```python
from src.ocr.engines import OpenOCREngine

engine = OpenOCREngine({
    'model': 'openocr-v1',  # ou 'openocr-v2'
    'device': 'cuda'         # ou 'cpu'
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### Configuração YAML
```yaml
# config/ocr/openocr.yaml
engine: openocr
model: openocr-v1
device: cuda

preprocessing:
  # OpenOCR funciona melhor com:
  - grayscale: true
  - clahe: true
  - denoise: true
```

#### Quando Usar
- ✅ Produção (recomendado)
- ✅ Textos em embalagens
- ✅ Múltiplos idiomas
- ✅ Textos com perspectiva

#### Quando NÃO Usar
- ❌ Hardware limitado (sem GPU)
- ❌ Necessita velocidade máxima

---

### 2. PARSeq Enhanced ⭐

**Melhor alternativa ao OpenOCR**

#### Características
- Arquitetura: Pure Transformer (encoder-decoder)
- Training: Permutation Language Modeling
- Idiomas: Multi-language
- GPU: Recomendado

#### Diferenciais
✅ Excelente para textos curtos (datas)  
✅ Entende contexto bidirecional  
✅ Múltiplos modelos (tiny, small, base)  
✅ Fine-tuned para datas  

#### Instalação
```bash
# PARSeq é carregado via torch.hub
# Sem instalação adicional necessária
```

#### Uso
```python
from src.ocr.engines import EnhancedPARSeqEngine

engine = EnhancedPARSeqEngine({
    'model': 'parseq_tiny',  # tiny/small/base
    'device': 'cuda'
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### Modelos Disponíveis
```yaml
parseq_tiny:   # 1.9M params, mais rápido
  speed: ⭐⭐⭐⭐⭐
  accuracy: ⭐⭐⭐⭐

parseq_small:  # 7.4M params, balanceado
  speed: ⭐⭐⭐⭐
  accuracy: ⭐⭐⭐⭐⭐

parseq_base:   # 23M params, melhor precisão
  speed: ⭐⭐⭐
  accuracy: ⭐⭐⭐⭐⭐
```

#### Quando Usar
- ✅ Alternativa ao OpenOCR
- ✅ Textos curtos (datas, códigos)
- ✅ Fine-tuning disponível
- ✅ Controle de tamanho do modelo

---

### 3. TrOCR

**Transformer-based OCR da Microsoft**

#### Características
- Arquitetura: Transformer Encoder-Decoder
- Pre-training: Masked Language Modeling
- Base: ViT (Vision Transformer)
- GPU: Obrigatório

#### Diferenciais
✅ Transfer learning (pré-treinado)  
✅ Alta capacidade de generalização  
✅ Bom para fine-tuning  
⚠️ Mais lento que outras opções  

#### Instalação
```bash
pip install transformers
```

#### Uso
```python
from src.ocr.engines import TrOCREngine

engine = TrOCREngine({
    'model': 'microsoft/trocr-base-printed',  # ou 'trocr-large-printed'
    'device': 'cuda'
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### Modelos Disponíveis
```yaml
trocr-base-printed:    # Textos impressos
trocr-large-printed:   # Melhor precisão
trocr-base-handwritten: # Textos manuscritos
```

#### Quando Usar
- ✅ Fine-tuning para domínio específico
- ✅ Pesquisa acadêmica
- ✅ Textos complexos
- ❌ NÃO para produção (muito lento)

---

### 4. EasyOCR

**OCR fácil e versátil**

#### Características
- Arquitetura: CRNN (CNN + LSTM)
- Idiomas: 80+ idiomas
- GPU: Opcional (funciona em CPU)
- Setup: Muito fácil

#### Diferenciais
✅ Facilidade de uso  
✅ Muitos idiomas suportados  
✅ Funciona sem GPU  
⚠️ Precisão moderada  

#### Instalação
```bash
pip install easyocr
```

#### Uso
```python
from src.ocr.engines import EasyOCREngine

engine = EasyOCREngine({
    'languages': ['pt', 'en'],
    'gpu': True  # ou False para CPU
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### Quando Usar
- ✅ Prototipagem rápida
- ✅ Sem GPU disponível
- ✅ Múltiplos idiomas raros
- ❌ NÃO quando precisão é crítica

---

### 5. PaddleOCR

**OCR da Baidu**

#### Características
- Arquitetura: PP-OCR (otimizada)
- Idiomas: Foco em chinês, mas suporta PT
- GPU: Opcional
- Velocidade: Muito rápida

#### Diferenciais
✅ Muito rápido  
✅ Otimizado para produção  
✅ Bom para textos orientais  
⚠️ Precisão menor em PT/EN  

#### Instalação
```bash
pip install paddlepaddle paddleocr
```

#### Uso
```python
from src.ocr.engines import PaddleOCREngine

engine = PaddleOCREngine({
    'lang': 'pt',
    'use_gpu': True
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### Quando Usar
- ✅ Velocidade é prioridade
- ✅ Textos em chinês/japonês/coreano
- ❌ NÃO para máxima precisão em PT

---

### 6. Tesseract

**OCR tradicional open-source**

#### Características
- Tipo: OCR tradicional (não deep learning)
- Idiomas: 100+ idiomas
- GPU: Não usa
- Velocidade: Muito rápido

#### Diferenciais
✅ Muito rápido (CPU)  
✅ Sem dependências de GPU  
✅ Leve (< 10MB)  
⚠️ Baixa precisão em textos difíceis  

#### Instalação
```bash
# Windows (via Chocolatey)
choco install tesseract

# Linux
sudo apt-get install tesseract-ocr

# Python
pip install pytesseract
```

#### Uso
```python
from src.ocr.engines import TesseractEngine

engine = TesseractEngine({
    'lang': 'por+eng',
    'config': '--psm 6 --oem 3'
})

engine.initialize()
text, confidence = engine.extract_text(image)
```

#### PSM (Page Segmentation Mode)
```
--psm 6   # Assume um bloco uniforme de texto (padrão)
--psm 7   # Trata a imagem como uma única linha de texto
--psm 8   # Trata a imagem como uma única palavra
--psm 13  # Raw line (sem layout)
```

#### Quando Usar
- ✅ Testes rápidos
- ✅ Hardware limitado (Raspberry Pi)
- ✅ Baseline para comparação
- ❌ NÃO para produção crítica

---

## 📊 Comparação Detalhada

### Uso de Memória

| Engine | GPU VRAM | RAM |
|--------|----------|-----|
| OpenOCR | 2.5 GB | 1 GB |
| PARSeq Tiny | 0.5 GB | 0.5 GB |
| PARSeq Base | 1.5 GB | 1 GB |
| TrOCR Base | 2 GB | 1.5 GB |
| TrOCR Large | 4 GB | 2 GB |
| EasyOCR | 1.5 GB | 1 GB |
| PaddleOCR | 1 GB | 0.5 GB |
| Tesseract | - | 0.2 GB |

---

## 🎯 Como Escolher

### Árvore de Decisão

```
Precisa de máxima precisão?
├─ SIM: OpenOCR ⭐
└─ NÃO
   │
   Tem GPU disponível?
   ├─ SIM
   │  │
   │  Velocidade é crítica?
   │  ├─ SIM: PARSeq Tiny
   │  └─ NÃO: PARSeq Enhanced
   │
   └─ NÃO
      │
      Pode aceitar precisão moderada?
      ├─ SIM: EasyOCR
      └─ NÃO: Tesseract (baseline)
```

### Por Caso de Uso

#### Produção (Alta Precisão)
1. **OpenOCR** (primeira escolha)
2. PARSeq Enhanced (alternativa)
3. TrOCR (se GPU grande)

#### Produção (Alta Velocidade)
1. **PARSeq Tiny** (com GPU)
2. PaddleOCR (com/sem GPU)
3. Tesseract (sem GPU)

#### Pesquisa/Experimentação
1. **TrOCR** (fine-tuning)
2. PARSeq Base (experimentação)
3. OpenOCR (baseline)

#### Hardware Limitado
1. **Tesseract** (sem GPU)
2. EasyOCR (CPU mode)
3. PaddleOCR (CPU mode)

---

## ⚙️ Uso e Configuração

### Interface Unificada

Todos os engines seguem a mesma interface:

```python
from src.ocr.engines import [Engine]

# 1. Criar engine
engine = [Engine](config)

# 2. Inicializar
engine.initialize()

# 3. Extrair texto
text, confidence = engine.extract_text(image)

# 4. Verificar disponibilidade
if engine.is_available():
    print(f"{engine.get_name()} disponível!")
```

### Configuração via YAML

```yaml
# config/ocr/meu_engine.yaml
engine: openocr  # ou parseq_enhanced, trocr, etc.

# Configurações específicas do engine
model: openocr-v1
device: cuda

# Pré-processamento
preprocessing:
  steps:
    grayscale:
      enabled: true
    clahe:
      enabled: true
      clipLimit: 2.0
    denoise:
      enabled: true
      h: 10
```

### Trocar Engine

```bash
# Via comando Make
make ocr-test ENGINE=openocr

# Via código Python
from src.ocr.engines import get_engine

engine = get_engine('openocr', config)
text, conf = engine.extract_text(image)
```

### Comparar Engines

```bash
# Compara todos em uma imagem
make ocr-compare

# Benchmark completo
make ocr-benchmark
```

---

## 🐛 Troubleshooting

### Erro: "CUDA out of memory"

**Causa**: Modelo muito grande para GPU  
**Solução**:
```yaml
device: cpu  # Forçar CPU
# ou
model: parseq_tiny  # Modelo menor
```

### Erro: "No module named 'openocr'"

**Causa**: Engine não instalado  
**Solução**:
```bash
pip install openocr
# ou
make ocr-setup  # Instala todos
```

### OCR retorna texto vazio

**Causas possíveis**:
1. Imagem muito escura/clara
2. Texto muito pequeno
3. Perspectiva extrema

**Soluções**:
```bash
# Tente outro engine
make ocr-test ENGINE=parseq_enhanced

# Ajuste pré-processamento
# Edite config/preprocessing/*.yaml
```

### Baixa precisão

**Soluções**:
1. Troque para OpenOCR (melhor precisão)
2. Ajuste pré-processamento
3. Verifique qualidade da imagem
4. Use ensemble (múltiplos engines)

---

## 📚 Recursos Adicionais

- **[Comparação de Modelos](15-MODEL-COMPARISON.md)**
- **[Pré-processamento](09-PREPROCESSING.md)**
- **[Otimização](21-OPTIMIZATION.md)**

---

**Anterior: [← Detecção YOLO](07-YOLO-DETECTION.md) | Próximo: [Pré-processamento →](09-PREPROCESSING.md)**
