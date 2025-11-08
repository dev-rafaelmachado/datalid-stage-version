# 📅 Pós-processamento e Parse de Datas

> Validação, normalização e extração inteligente de datas

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Parse de Datas](#parse-de-datas)
- [Validação](#validação)
- [Fuzzy Matching](#fuzzy-matching)
- [Normalização](#normalização)
- [Scoring e Seleção](#scoring-e-seleção)

## 🎯 Visão Geral

O pós-processamento transforma **texto bruto do OCR** em **datas estruturadas e validadas**.

**Desafios:**
- 📝 OCR pode ter erros: "15i03i2025" ao invés de "15/03/2025"
- 🌍 Múltiplos formatos: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
- 🏷️ Prefixos/sufixos: "VAL 15/03/2025", "Validade: 15-03-2025"
- 🔢 Números ambíguos: "01/02/2025" = 1º Fev ou 2 Jan?
- ❌ Datas inválidas: 32/13/2025, 29/02/2023

**Solução do Datalid:**
- ✅ Parse multi-formato com regex
- ✅ Validação de lógica de datas
- ✅ Fuzzy matching para correção de erros
- ✅ Scoring inteligente para múltiplas candidatas
- ✅ Normalização para formato padrão

## 📅 Parse de Datas

### Formatos Suportados

```python
# config/pipeline/full_pipeline.yaml
parsing:
  date_formats:
    # DD/MM/YYYY (Padrão BR)
    - "%d/%m/%Y"
    - "%d-%m-%Y"
    - "%d.%m.%Y"
    - "%d %m %Y"
    
    # MM/DD/YYYY (US)
    - "%m/%d/%Y"
    - "%m-%d-%Y"
    
    # YYYY-MM-DD (ISO)
    - "%Y-%m-%d"
    - "%Y/%m/%d"
    
    # Variações com 2 dígitos para ano
    - "%d/%m/%y"
    - "%d-%m-%y"
    
    # Com texto
    - "VAL %d/%m/%Y"
    - "Validade %d/%m/%Y"
    - "Venc %d/%m/%Y"
    - "EXP %d/%m/%Y"
```

### Extração com Regex

```python
from src.ocr.postprocessors import DateParser

parser = DateParser(config)

# Texto do OCR
text = "VAL 15/03/2025"

# Extrair padrões de data
patterns = parser.extract_date_patterns(text)
# Output: ["15/03/2025", "15-03-2025", "15.03.2025"]
```

**Regex Patterns Usados:**

```python
DATE_PATTERNS = [
    # DD/MM/YYYY
    r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b',
    
    # DD/MM/YY
    r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})\b',
    
    # YYYY-MM-DD
    r'\b(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b',
    
    # DD MM YYYY (com espaços)
    r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b',
    
    # Com prefixos comuns
    r'(?:VAL|VALIDADE|VENC|EXP|EXPIRA)\s*[:.]?\s*(\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{2,4})',
]
```

### Parse e Conversão

```python
from datetime import datetime

def parse_date(text, formats):
    """Parse text to datetime object."""
    for fmt in formats:
        try:
            date_obj = datetime.strptime(text, fmt)
            return {
                'date': date_obj,
                'format': fmt,
                'text': text
            }
        except ValueError:
            continue
    return None

# Exemplo
result = parse_date("15/03/2025", ["%d/%m/%Y"])
# {
#     'date': datetime(2025, 3, 15),
#     'format': '%d/%m/%Y',
#     'text': '15/03/2025'
# }
```

## ✅ Validação

### Validações Aplicadas

#### 1. Validação de Formato

```python
def validate_format(day, month, year):
    """Valida ranges de dia/mês/ano."""
    if not (1 <= day <= 31):
        return False, "Dia inválido"
    
    if not (1 <= month <= 12):
        return False, "Mês inválido"
    
    if not (2020 <= year <= 2050):
        return False, "Ano fora do range esperado"
    
    return True, "OK"
```

#### 2. Validação de Existência

```python
def validate_existence(date_obj):
    """Verifica se a data realmente existe."""
    try:
        # datetime já faz essa validação
        # Ex: 29/02/2023 = ValueError
        datetime(date_obj.year, date_obj.month, date_obj.day)
        return True, "Data válida"
    except ValueError as e:
        return False, str(e)
```

#### 3. Validação de Razoabilidade

```python
def validate_reasonableness(date_obj, today=None):
    """Valida se a data é razoável para validade."""
    if today is None:
        today = datetime.now()
    
    # Data no passado (expirada)
    if date_obj < today:
        days_ago = (today - date_obj).days
        if days_ago > 365:  # Mais de 1 ano no passado
            return False, f"Data muito antiga ({days_ago} dias atrás)"
        return True, "Expirada"
    
    # Data no futuro
    days_ahead = (date_obj - today).days
    
    # Muito distante no futuro
    if days_ahead > 3650:  # 10 anos
        return False, f"Data muito distante ({days_ahead} dias)"
    
    return True, "OK"
```

#### 4. Validação de Ambiguidade

```python
def resolve_ambiguity(text, formats):
    """
    Resolve ambiguidade DD/MM vs MM/DD.
    
    Ex: "01/12/2025" pode ser:
    - 1º de Dezembro (DD/MM)
    - 12 de Janeiro (MM/DD)
    """
    candidates = []
    
    for fmt in formats:
        try:
            date_obj = datetime.strptime(text, fmt)
            
            # Preferir DD/MM para Brasil
            if fmt == "%d/%m/%Y":
                priority = 2
            elif fmt == "%m/%d/%Y":
                priority = 1
            else:
                priority = 0
            
            candidates.append({
                'date': date_obj,
                'format': fmt,
                'priority': priority
            })
        except ValueError:
            continue
    
    # Retornar candidato com maior prioridade
    if candidates:
        return max(candidates, key=lambda x: x['priority'])
    
    return None
```

### Exemplo Completo de Validação

```python
def validate_date(text, date_obj, config):
    """Valida uma data parsed."""
    checks = {
        'format_valid': False,
        'date_exists': False,
        'is_reasonable': False,
        'is_future': False,
        'errors': []
    }
    
    # 1. Validar formato
    valid, msg = validate_format(
        date_obj.day, 
        date_obj.month, 
        date_obj.year
    )
    checks['format_valid'] = valid
    if not valid:
        checks['errors'].append(msg)
        return checks
    
    # 2. Validar existência
    valid, msg = validate_existence(date_obj)
    checks['date_exists'] = valid
    if not valid:
        checks['errors'].append(msg)
        return checks
    
    # 3. Validar razoabilidade
    valid, msg = validate_reasonableness(date_obj)
    checks['is_reasonable'] = valid
    if not valid:
        checks['errors'].append(msg)
    
    # 4. Verificar se está no futuro
    today = datetime.now()
    checks['is_future'] = date_obj > today
    
    # 5. Calcular dias até expiração
    if checks['is_future']:
        checks['days_until_expiry'] = (date_obj - today).days
        
        # Classificar status
        if checks['days_until_expiry'] <= 7:
            checks['status'] = 'near_expiry'
        elif checks['days_until_expiry'] <= 30:
            checks['status'] = 'warning'
        else:
            checks['status'] = 'valid'
    else:
        checks['days_until_expiry'] = (today - date_obj).days * -1
        checks['status'] = 'expired'
    
    return checks
```

## 🔍 Fuzzy Matching

### Correção de Erros de OCR

O OCR frequentemente confunde caracteres similares:

| Correto | OCR Lê | Confusão |
|---------|--------|----------|
| 0 | O | Letra O |
| 1 | I, l | Letra I ou L |
| 5 | S | Letra S |
| 8 | B | Letra B |
| / | i, l, 1 | Caracteres similares |

### Implementação

```python
import re
from difflib import SequenceMatcher

def fuzzy_fix_date(text, threshold=0.8):
    """
    Tenta corrigir erros comuns de OCR em datas.
    """
    # 1. Substituições diretas
    fixes = {
        'O': '0',  # O → 0
        'o': '0',
        'I': '1',  # I → 1
        'l': '1',
        'i': '/',  # i → /
        'S': '5',  # S → 5
        'B': '8',  # B → 8
        ' ': '',   # Remover espaços extras
    }
    
    fixed = text
    for wrong, right in fixes.items():
        fixed = fixed.replace(wrong, right)
    
    # 2. Padronizar separadores
    fixed = re.sub(r'[^\d]+', '/', fixed)  # Tudo que não é dígito → /
    
    # 3. Validar estrutura básica DD/MM/YYYY
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', fixed)
    if match:
        day, month, year = match.groups()
        return f"{day}/{month}/{year}"
    
    return None

# Exemplo
text_ocr = "15i03i2O25"  # OCR com erros
fixed = fuzzy_fix_date(text_ocr)
print(fixed)  # "15/03/2025"
```

### Matching com Levenshtein Distance

```python
def levenshtein_distance(s1, s2):
    """Calcula distância de edição entre strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Custo de inserção, deleção ou substituição
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def fuzzy_match_date(text, known_formats, threshold=0.85):
    """
    Encontra melhor match fuzzy com formatos conhecidos.
    """
    best_match = None
    best_score = 0
    
    for fmt in known_formats:
        # Exemplo de formato: "DD/MM/YYYY"
        pattern = fmt.replace('DD', r'\d{2}')
        pattern = pattern.replace('MM', r'\d{2}')
        pattern = pattern.replace('YYYY', r'\d{4}')
        
        # Calcular similaridade
        score = SequenceMatcher(None, text, pattern).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = fmt
    
    return best_match, best_score
```

## 🔄 Normalização

### Normalização de Output

```python
def normalize_date(date_obj, output_format='%d/%m/%Y'):
    """
    Normaliza data para formato padrão.
    """
    return {
        'date': date_obj.strftime(output_format),
        'day': date_obj.day,
        'month': date_obj.month,
        'year': date_obj.year,
        'iso': date_obj.isoformat(),  # YYYY-MM-DD
        'timestamp': date_obj.timestamp(),
        'weekday': date_obj.strftime('%A'),
        'month_name': date_obj.strftime('%B')
    }

# Exemplo
date_obj = datetime(2025, 3, 15)
normalized = normalize_date(date_obj)
# {
#     'date': '15/03/2025',
#     'day': 15,
#     'month': 3,
#     'year': 2025,
#     'iso': '2025-03-15',
#     'timestamp': 1742083200.0,
#     'weekday': 'Saturday',
#     'month_name': 'March'
# }
```

## 🏆 Scoring e Seleção

### Sistema de Scoring

Quando múltiplas datas são encontradas, precisamos selecionar a melhor:

```python
def score_date_candidate(candidate, ocr_confidence, config):
    """
    Calcula score composto para uma candidata.
    
    Fatores:
    - Confiança do OCR (40%)
    - Formato comum (20%)
    - Razoabilidade da data (20%)
    - Consistência com outras detecções (20%)
    """
    score = 0.0
    weights = config.get('scoring_weights', {
        'ocr_confidence': 0.4,
        'format_common': 0.2,
        'date_reasonable': 0.2,
        'consistency': 0.2
    })
    
    # 1. Confiança do OCR
    score += weights['ocr_confidence'] * ocr_confidence
    
    # 2. Formato comum
    common_formats = [
        '%d/%m/%Y',  # BR padrão
        '%d-%m-%Y',
        '%Y-%m-%d'   # ISO
    ]
    if candidate['format'] in common_formats:
        score += weights['format_common']
    
    # 3. Razoabilidade
    days = candidate.get('days_until_expiry', 0)
    if 0 < days < 730:  # Entre hoje e 2 anos
        # Preferir datas entre 1 mês e 1 ano
        if 30 < days < 365:
            date_score = 1.0
        elif days <= 30:
            date_score = 0.8  # Muito próximo (possível erro)
        else:
            date_score = 0.6  # Muito distante
        
        score += weights['date_reasonable'] * date_score
    
    # 4. Consistência (se múltiplas detecções)
    # Se a mesma data aparece várias vezes, é mais confiável
    occurrence_count = candidate.get('occurrence_count', 1)
    max_occurrences = candidate.get('max_occurrences', 1)
    consistency_score = occurrence_count / max_occurrences
    score += weights['consistency'] * consistency_score
    
    return score
```

### Seleção da Melhor Data

```python
def select_best_date(candidates, config):
    """
    Seleciona a melhor data entre candidatas.
    """
    if not candidates:
        return None
    
    # Calcular score para cada candidata
    scored = []
    for candidate in candidates:
        score = score_date_candidate(
            candidate,
            candidate['ocr_confidence'],
            config
        )
        candidate['score'] = score
        scored.append(candidate)
    
    # Ordenar por score (maior primeiro)
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    # Retornar melhor
    best = scored[0]
    
    # Log de decisão
    logger.info(f"Melhor data selecionada: {best['date']}")
    logger.info(f"  Score: {best['score']:.3f}")
    logger.info(f"  OCR Confidence: {best['ocr_confidence']:.2%}")
    logger.info(f"  Formato: {best['format']}")
    
    # Retornar também alternativas (top 3)
    return {
        'best': best,
        'alternatives': scored[1:4]  # Top 2-4
    }
```

### Exemplo Completo

```python
from src.ocr.postprocessors import DateParser

# Configuração
config = {
    'date_formats': [
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%m/%d/%Y'
    ],
    'fuzzy_matching': True,
    'fuzzy_threshold': 0.85,
    'scoring_weights': {
        'ocr_confidence': 0.4,
        'format_common': 0.2,
        'date_reasonable': 0.2,
        'consistency': 0.2
    }
}

# Inicializar parser
parser = DateParser(config)

# OCR results (múltiplas detecções)
ocr_results = [
    {'text': 'VAL 15/03/2025', 'confidence': 0.92},
    {'text': '15i03i2025', 'confidence': 0.85},  # Com erro
    {'text': 'Validade 15-03-2025', 'confidence': 0.88}
]

# Parse todas
all_dates = []
for ocr_result in ocr_results:
    parsed = parser.parse(
        ocr_result['text'],
        ocr_confidence=ocr_result['confidence']
    )
    if parsed:
        all_dates.extend(parsed)

# Selecionar melhor
result = parser.select_best(all_dates)

print(f"Melhor data: {result['best']['date']}")
print(f"Confiança final: {result['best']['score']:.2%}")
print(f"Dias até expirar: {result['best']['days_until_expiry']}")
```

## 📊 Métricas

### Avaliação de Performance

```python
def evaluate_parsing(predictions, ground_truth):
    """Avalia precisão do parsing."""
    correct = 0
    total = len(ground_truth)
    
    for pred, gt in zip(predictions, ground_truth):
        if pred['date'] == gt['date']:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
```

## 📚 Referências

- [Python datetime](https://docs.python.org/3/library/datetime.html)
- [Regex for dates](https://regex101.com/)
- [Fuzzy string matching](https://github.com/seatgeek/fuzzywuzzy)

## 💡 Próximos Passos

- **[Pipeline Completo](11-FULL-PIPELINE.md)** - Integração end-to-end
- **[Avaliação](14-EVALUATION.md)** - Métricas de performance
- **[API REST](16-API-REST.md)** - Usar em produção

---

**Dúvidas sobre parsing?** Consulte [FAQ](25-FAQ.md) ou [Troubleshooting](22-TROUBLESHOOTING.md)
