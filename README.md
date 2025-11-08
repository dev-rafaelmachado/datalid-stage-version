# Datalid 3.0

Sistema modular para detecção e extração de datas de validade em imagens, combinando detecção/segmentação (YOLO) com pipelines OCR e pós-processamento especializado para datas.

## Objetivo
- Fornecer um pipeline robusto, configurável e fácil de integrar para localizar regiões candidatas e extrair informações de datas com confiança.

## Visão geral (essencial)
- Detector/segmentador (YOLO) identifica regiões relevantes.
- Normalização e/ou segmentação de linhas para melhorar entrada do OCR.
- Engines OCR configuráveis (PARSeq, TrOCR, Tesseract, OpenOCR, EasyOCR, etc.).
- Pós-processamento: validação, parsing e heurísticas específicas para datas.

## Uso mínimo necessário
1. Instalar dependências: veja `requirements.txt`.
2. Rodar inferência em uma imagem (exemplo mínimo):
   - scripts de inferência: `scripts/inference/predict_single.py` (aponta imagem e modelo).
3. Ajustes rápidos: altere presets e pipelines em `config/` e `config/pipeline/`.

## Estrutura principal
- `src/` — código-fonte principal (yolo, ocr, pipeline, utils).
- `scripts/` — utilitários para inferência, treinamento, avaliação e preparação de dados.
- `config/` — configurações e presets (engines, pipelines, experimentos).
- `data/` — imagens, datasets e resultados amostra.
- `docs/` — documentação técnica (arquitetura, avaliação, pré-processamento, etc.).

## Configuração e extensibilidade (rápido)
- Comportamento guiado por YAML em `config/` e `config/ocr/`.
- Componentes são modulares: troque a engine OCR ou o modelo YOLO via configs e presets.
- Experimentos reproduzíveis em `experiments/` (presets / args.yaml).

## Avaliação da Pipeline 📊
Sistema completo de avaliação end-to-end (YOLO → OCR → Parsing):

```bash
# Teste rápido em uma imagem
make pipeline-test IMAGE=data/sample.jpg

# Avaliação rápida (10 imagens)
make pipeline-eval-quick

# Avaliação customizada
make pipeline-eval NUM=20 MODE=random

# Avaliação completa (todas as imagens)
make pipeline-eval-full
```

**Métricas calculadas:**
- Detecção (YOLO): taxa de detecção, confiança média
- OCR: exact match, CER, WER, similaridade
- Parsing: taxa de datas encontradas
- End-to-end: acurácia da pipeline, tempo de processamento

**Outputs gerados:**
- CSV detalhado com resultados por imagem
- Métricas agregadas (JSON)
- Relatório markdown formatado
- Visualizações e gráficos
- Análise de erros por etapa

Veja [`docs/PIPELINE_EVALUATION_QUICK.md`](docs/PIPELINE_EVALUATION_QUICK.md) para guia rápido ou [`docs/PIPELINE_EVALUATION.md`](docs/PIPELINE_EVALUATION.md) para documentação completa.

## Onde olhar primeiro
- `docs/ARCHITECTURE.md` — visão técnica resumida do fluxo e decisões de design.
- `scripts/inference/predict_single.py` — ponto de entrada para inferência rápida.
- `config/project_config.yaml` e `config/pipeline/full_pipeline.yaml` — configuração do pipeline padrão.
- `docs/PIPELINE_EVALUATION_QUICK.md` — guia rápido de avaliação da pipeline.

## Contribuição e contato
- Abra uma issue para bugs ou sugestões.
- Mantenha alterações na pasta `experiments/` e `config/` para reprodutibilidade.

## Licença
- Verifique o arquivo de licença (adicionar se ausente).

## 🌐 API REST

**Nova funcionalidade!** Sistema completo de API REST para integração com outros aplicativos.

### Início Rápido

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar API
make api-run
# ou
python scripts/api/run_api.py

# API disponível em: http://localhost:8000
```

### Usar a API

**Python:**
```python
from scripts.api.client import DatalidClient

client = DatalidClient("http://localhost:8000")
result = client.process_image("produto.jpg")
print(f"Data: {result['best_date']['date']}")
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

fetch('http://localhost:8000/process', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log('Data:', data.best_date.date));
```

### Recursos

- ✅ **Endpoints RESTful** com documentação automática (Swagger/ReDoc)
- ✅ **Múltiplas Engines OCR** (OpenOCR, Tesseract, EasyOCR, PaddleOCR, PARSeq, TrOCR)
- ✅ **Processamento em Lote** para múltiplas imagens
- ✅ **Rate Limiting** e autenticação opcional
- ✅ **Docker Support** para deploy fácil
- ✅ **Cliente Python** incluído
- ✅ **Frontend Demo** interativo

### Documentação

- [**Guia Rápido**](docs/API_QUICK_START.md) - Comece em 5 minutos
- [**Documentação Completa**](docs/API.md) - Guia detalhado
- **Swagger UI**: http://localhost:8000/docs (após iniciar)
- **ReDoc**: http://localhost:8000/redoc

### Comandos Make

```bash
make api-run          # Iniciar API
make api-dev          # Modo desenvolvimento (auto-reload)
make api-test         # Testar todos os endpoints
make api-health       # Verificar status
make api-docker-build # Build Docker
make api-compose-up   # Docker Compose
```

### Frontend Demo

Abra `examples/frontend_demo.html` no navegador para interface web interativa.
