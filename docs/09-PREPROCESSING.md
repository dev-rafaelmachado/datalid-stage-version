# 🖼️ Pré-processamento de Imagens

> Técnicas de otimização de imagens para maximizar a precisão do OCR

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Pipeline de Pré-processamento](#pipeline-de-pré-processamento)
- [Técnicas Disponíveis](#técnicas-disponíveis)
- [Configurações](#configurações)
- [Quando Usar](#quando-usar)
- [Casos de Uso](#casos-de-uso)

## 🎯 Visão Geral

O pré-processamento é **crucial** para o sucesso do OCR. Uma imagem bem preparada pode aumentar a precisão de 60% para 95%+.

**Objetivos:**
- 🎯 **Melhorar contraste** entre texto e fundo
- 🔍 **Aumentar nitidez** de caracteres
- 🧹 **Remover ruído** visual
- 📐 **Corrigir perspectiva** e inclinação
- 📏 **Normalizar** tamanho e iluminação

**Quando aplicar:**
- ✅ **Sempre** para imagens reais (fotos de produtos)
- ⚠️ **Opcional** para imagens sintéticas/digitais
- ❌ **Nunca** exagere - pode piorar o resultado

## 🔄 Pipeline de Pré-processamento

```
┌─────────────────┐
│  Crop Original  │  RGB, (165x55x3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Resize      │  Manter legibilidade (max_height: 64px)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Deskew      │  Corrigir inclinação/rotação
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Normalize   │  Normalizar canais de cor
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. CLAHE       │  Equalização de histograma adaptativa
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Denoise     │  Remoção de ruído (bilateral filter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Sharpen     │  Aumentar nitidez dos caracteres
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. Binarize    │  Threshold adaptativo (Otsu)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Image Final    │  Otimizada para OCR
└─────────────────┘
```

## 🛠️ Técnicas Disponíveis

### 1. Resize (Redimensionamento)

**Objetivo:** Padronizar tamanho mantendo legibilidade

```python
from src.ocr.preprocessors import ImagePreprocessor

preprocessor = ImagePreprocessor(config)

# Resize mantendo aspect ratio
resized = preprocessor.resize_image(
    image,
    max_height=64,  # Altura máxima
    maintain_aspect=True
)
```

**Parâmetros:**
```yaml
preprocessing:
  resize:
    enable: true
    max_height: 64      # Altura máxima (pixels)
    max_width: null     # Largura máxima (null = auto)
    maintain_aspect: true
    interpolation: INTER_CUBIC  # INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
```

**Quando usar:**
- ✅ Imagens muito grandes (> 200px altura)
- ✅ Padronizar inputs para OCR
- ❌ Não reduza demais (mín. 32px altura)

### 2. Deskew (Correção de Inclinação)

**Objetivo:** Corrigir texto inclinado

```python
# Detectar ângulo de inclinação
angle = preprocessor.detect_skew(image)
print(f"Inclinação detectada: {angle}°")

# Corrigir
deskewed = preprocessor.deskew_image(image, angle)
```

**Parâmetros:**
```yaml
preprocessing:
  deskew:
    enable: true
    angle_threshold: 1.0  # Corrigir se |angle| > 1°
    bg_color: [255, 255, 255]  # Cor de preenchimento
```

**Quando usar:**
- ✅ Fotos tiradas em ângulo
- ✅ Texto visivelmente inclinado
- ❌ Já alinhado (perde tempo)

### 3. Normalize (Normalização)

**Objetivo:** Padronizar iluminação e cores

```python
# Normalizar canais RGB
normalized = preprocessor.normalize_image(image)
```

**Parâmetros:**
```yaml
preprocessing:
  normalize:
    enable: true
    method: minmax  # minmax, zscore, clahe
```

**Implementação:**
```python
def normalize_image(image):
    """Normalizar canais para [0, 255]."""
    # Min-max normalization
    normalized = cv2.normalize(
        image, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX
    )
    return normalized
```

**Quando usar:**
- ✅ Imagens com iluminação irregular
- ✅ Cores muito saturadas ou apagadas
- ✅ Sempre (baixo custo, alto benefício)

### 4. CLAHE (Equalização de Histograma)

**Objetivo:** Melhorar contraste local

```python
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe_image = preprocessor.apply_clahe(
    image,
    clip_limit=2.0,  # Limita amplificação de ruído
    tile_size=8      # Tamanho da grade
)
```

**Parâmetros:**
```yaml
preprocessing:
  clahe:
    enable: true
    clip_limit: 2.0   # 1.0-4.0 (menor = mais natural)
    tile_size: 8      # 8x8 ou 16x16
```

**Comparação:**

| Método | Descrição | Uso |
|--------|-----------|-----|
| **Equalização Global** | Histograma global | Imagens uniformes |
| **CLAHE** | Equalização por regiões | **Recomendado** (iluminação irregular) |

**Visualização:**
```
Original:          CLAHE:
▓▓▓░░░░░░░   →   ▓▓▓▒▒░░░░░
▓▓▓░░░░░░░        ▓▓▓▒▒░░░░░
▓▓▓░░░░░░░        ▓▓▓▒▒░░░░░
(baixo contraste)  (alto contraste)
```

**Quando usar:**
- ✅ Iluminação irregular
- ✅ Sombras parciais
- ✅ Reflexos localizados
- ❌ Já tem bom contraste

### 5. Denoise (Remoção de Ruído)

**Objetivo:** Remover artefatos e granulosidade

```python
# Bilateral filter (preserva bordas)
denoised = preprocessor.denoise_image(
    image,
    strength=5,  # 1-10
    method='bilateral'
)
```

**Parâmetros:**
```yaml
preprocessing:
  denoise:
    enable: true
    method: bilateral  # bilateral, gaussian, nlm
    strength: 5        # 1 (suave) - 10 (agressivo)
```

**Métodos disponíveis:**

| Método | Velocidade | Qualidade | Uso |
|--------|-----------|-----------|-----|
| **Gaussian** | ⚡⚡⚡ | ⭐⭐ | Ruído uniforme |
| **Bilateral** | ⚡⚡ | ⭐⭐⭐ | **Recomendado** (preserva bordas) |
| **Non-Local Means** | ⚡ | ⭐⭐⭐⭐ | Máxima qualidade (lento) |

**Implementação Bilateral:**
```python
denoised = cv2.bilateralFilter(
    image,
    d=5,           # Diameter
    sigmaColor=75,  # Color space sigma
    sigmaSpace=75   # Coordinate space sigma
)
```

**Quando usar:**
- ✅ Imagens com grão/ruído
- ✅ Fotos em baixa luz
- ✅ Compression artifacts
- ❌ Imagens limpas (pode borrar)

### 6. Sharpen (Aumento de Nitidez)

**Objetivo:** Realçar bordas dos caracteres

```python
# Unsharp masking
sharpened = preprocessor.sharpen_image(
    image,
    amount=1.5,  # 0.5-3.0
    radius=1.0   # 0.5-2.0
)
```

**Parâmetros:**
```yaml
preprocessing:
  sharpen:
    enable: true
    amount: 1.5      # Intensidade
    radius: 1.0      # Raio do kernel
    threshold: 0     # Threshold (0 = tudo)
```

**Implementação (Unsharp Mask):**
```python
def sharpen_image(image, amount=1.5, radius=1.0):
    """Sharpening via unsharp masking."""
    # Blur
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    
    # Sharpened = Original + amount * (Original - Blurred)
    sharpened = cv2.addWeighted(
        image, 1.0 + amount,
        blurred, -amount,
        0
    )
    return sharpened
```

**Quando usar:**
- ✅ Texto levemente borrado
- ✅ Após denoise (pode borrar)
- ✅ Fotos fora de foco (leve)
- ❌ Já nítido (amplifica ruído)

### 7. Binarize (Binarização)

**Objetivo:** Converter para preto e branco (texto/fundo)

```python
# Threshold adaptativo
binary = preprocessor.binarize_image(
    image,
    method='adaptive',  # adaptive, otsu, simple
    block_size=11,
    c=2
)
```

**Parâmetros:**
```yaml
preprocessing:
  binarize:
    enable: true
    method: adaptive     # adaptive, otsu, simple
    block_size: 11       # Para adaptive (ímpar)
    c: 2                 # Constante subtraída
    invert: false        # Inverter (texto branco/fundo preto)
```

**Métodos:**

#### Adaptive Threshold (Recomendado)
```python
binary = cv2.adaptiveThreshold(
    gray_image,
    255,                           # Max value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,                  # Neighborhood size
    C=2                            # Constant
)
```

**Melhor para:** Iluminação irregular

#### Otsu's Method
```python
_, binary = cv2.threshold(
    gray_image,
    0,                    # Threshold (auto-calculado)
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

**Melhor para:** Iluminação uniforme

#### Simple Threshold
```python
_, binary = cv2.threshold(
    gray_image,
    127,                  # Threshold fixo
    255,
    cv2.THRESH_BINARY
)
```

**Melhor para:** Testes rápidos

**Comparação Visual:**
```
Original:       Adaptive:       Otsu:
████▓▓▓▒▒░░  →  ████████░░░  vs  ████░░░░░░░
```

**Quando usar:**
- ✅ Fundo complexo
- ✅ Cores similares (texto/fundo)
- ✅ Para engines OCR tradicionais (Tesseract)
- ❌ Para deep learning OCR (preferem grayscale/RGB)

## ⚙️ Configurações

### Preset: Mínimo (Rápido)

```yaml
# config/preprocessing/ppro-minimal.yaml
preprocessing:
  enable: true
  
  resize:
    enable: true
    max_height: 64
  
  normalize:
    enable: true
  
  clahe:
    enable: true
    clip_limit: 2.0
```

**Uso:** Imagens de boa qualidade

### Preset: Completo (Máxima Qualidade)

```yaml
# config/preprocessing/ppro-complete.yaml
preprocessing:
  enable: true
  
  resize:
    enable: true
    max_height: 64
  
  deskew:
    enable: true
  
  normalize:
    enable: true
  
  clahe:
    enable: true
    clip_limit: 3.0
  
  denoise:
    enable: true
    method: bilateral
    strength: 7
  
  sharpen:
    enable: true
    amount: 2.0
  
  binarize:
    enable: true
    method: adaptive
    block_size: 15
```

**Uso:** Imagens difíceis (baixa qualidade, ruído, iluminação ruim)

### Preset: OCR-Specific

Diferentes OCR engines se beneficiam de diferentes preprocessamentos:

#### Para Tesseract
```yaml
# config/preprocessing/ppro-tesseract.yaml
preprocessing:
  binarize:
    enable: true      # Tesseract prefere binário
    method: adaptive
```

#### Para Deep Learning (OpenOCR, PARSeq, TrOCR)
```yaml
# config/preprocessing/ppro-openocr.yaml
preprocessing:
  binarize:
    enable: false     # DL prefere grayscale/RGB
  
  clahe:
    enable: true      # Contraste é importante
  
  denoise:
    enable: true      # Remover artefatos
```

## 🎯 Quando Usar

### Matriz de Decisão

| Condição da Imagem | Resize | Deskew | Normalize | CLAHE | Denoise | Sharpen | Binarize |
|-------------------|--------|--------|-----------|-------|---------|---------|----------|
| **Boa qualidade** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Iluminação ruim** | ✅ | ❌ | ✅ | ✅✅ | ✅ | ❌ | ⚠️ |
| **Borrada** | ✅ | ❌ | ✅ | ✅ | ❌ | ✅✅ | ❌ |
| **Inclinada** | ✅ | ✅✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Ruidosa** | ✅ | ❌ | ✅ | ✅ | ✅✅ | ❌ | ❌ |
| **Baixo contraste** | ✅ | ❌ | ✅ | ✅✅ | ❌ | ✅ | ⚠️ |
| **Fundo complexo** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅✅ |

✅ = Recomendado  
✅✅ = Essencial  
⚠️ = Testar  
❌ = Não necessário

## 📊 Casos de Uso

### Caso 1: Foto de Celular (Qualidade Normal)

**Problema:** Iluminação razoável, mas pode ter leve blur

**Solução:**
```yaml
preprocessing:
  enable: true
  resize: {enable: true, max_height: 64}
  normalize: {enable: true}
  clahe: {enable: true, clip_limit: 2.0}
  sharpen: {enable: true, amount: 1.0}
```

**Resultado:** 85% → 92% precisão

### Caso 2: Imagem Escaneada (Alta Qualidade)

**Problema:** Boa, mas pode ter ruído de scanner

**Solução:**
```yaml
preprocessing:
  enable: true
  resize: {enable: true, max_height: 64}
  normalize: {enable: true}
  denoise: {enable: true, strength: 3}
```

**Resultado:** 90% → 95% precisão

### Caso 3: Foto em Baixa Luz

**Problema:** Escura, baixo contraste, ruidosa

**Solução:**
```yaml
preprocessing:
  enable: true
  resize: {enable: true, max_height: 64}
  normalize: {enable: true}
  clahe: {enable: true, clip_limit: 3.5}
  denoise: {enable: true, method: bilateral, strength: 8}
  sharpen: {enable: true, amount: 1.5}
```

**Resultado:** 60% → 88% precisão

### Caso 4: Texto Inclinado

**Problema:** Foto tirada em ângulo

**Solução:**
```yaml
preprocessing:
  enable: true
  deskew: {enable: true, angle_threshold: 1.0}
  resize: {enable: true, max_height: 64}
  normalize: {enable: true}
  clahe: {enable: true, clip_limit: 2.0}
```

**Resultado:** 70% → 91% precisão

### Caso 5: Fundo Complexo

**Problema:** Texto sobre textura/padrão

**Solução:**
```yaml
preprocessing:
  enable: true
  resize: {enable: true, max_height: 64}
  normalize: {enable: true}
  clahe: {enable: true, clip_limit: 3.0}
  denoise: {enable: true, strength: 6}
  binarize: {enable: true, method: adaptive, block_size: 15}
```

**Resultado:** 65% → 87% precisão

## 🧪 Experimentação

### Testar Diferentes Configurações

```bash
# Script de experimentação
python scripts/preprocessing/test_preprocessing.py \
  --image data/ocr_test/difficult_image.jpg \
  --configs config/preprocessing/*.yaml \
  --output outputs/preprocessing_tests/
```

**Output:**
```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Config           ┃ Text Extracted   ┃ CER       ┃ Time(s)  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ ppro-minimal     │ VAL 15/03/2025  │ 0.00      │ 0.05     │
│ ppro-complete    │ VAL 15/03/2025  │ 0.00      │ 0.18     │
│ ppro-tesseract   │ VAL 15/03/2025  │ 0.00      │ 0.12     │
└──────────────────┴──────────────────┴───────────┴──────────┘
```

### Visualizar Etapas

```python
from src.ocr.preprocessors import ImagePreprocessor
from src.ocr.visualization import visualize_preprocessing_steps

# Carregar imagem
image = cv2.imread('crop.jpg')

# Processar com visualização de etapas
preprocessor = ImagePreprocessor(config)
result = preprocessor.process_with_steps(image)

# result contém cada etapa:
# - original
# - resized
# - normalized
# - clahe
# - denoised
# - sharpened
# - binarized

# Criar visualização lado a lado
vis = visualize_preprocessing_steps(result)
cv2.imwrite('preprocessing_steps.jpg', vis)
```

## 🐛 Troubleshooting

### Problema: OCR Piorou Após Pré-processamento

**Causas:**
- Pré-processamento muito agressivo
- Binarização removeu informação importante
- Sharpen amplificou ruído

**Soluções:**
1. Reduzir intensidades (clip_limit, strength, amount)
2. Desabilitar binarização para DL models
3. Testar com pré-processamento mínimo
4. Usar preset específico para o OCR engine

### Problema: Muito Lento

**Soluções:**
```yaml
preprocessing:
  # Desabilitar etapas custosas
  denoise: {enable: false}  # Mais custoso
  deskew: {enable: false}   # Se não necessário
  binarize: {enable: false} # Se DL model
  
  # Reduzir resolução
  resize: {max_height: 48}  # Menor = mais rápido
```

### Problema: Texto Desapareceu na Binarização

**Solução:**
```yaml
binarize:
  method: adaptive    # Mais robusto que Otsu
  block_size: 21      # Aumentar (maior vizinhança)
  c: 1                # Reduzir (menos agressivo)
```

## 📚 Referências

- [OpenCV Preprocessing](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- [Image Denoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)

## 💡 Próximos Passos

- **[Sistema OCR](08-OCR-SYSTEM.md)** - Engines que usarão essas imagens
- **[Otimização](21-OPTIMIZATION.md)** - Melhorar performance
- **[Best Practices](24-BEST-PRACTICES.md)** - Padrões recomendados

---

**Dúvidas sobre pré-processamento?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
