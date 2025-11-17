# ========================================
# 🚀 Makefile - Datalid 3.0
# Sistema de Detecção de Datas de Validade
# ========================================

# Configurações
PYTHON := python
PIP := pip
PROJECT_NAME := datalid
VERSION := 3.0.0

# Caminhos
SRC_DIR := src
SCRIPTS_DIR := scripts
DATA_DIR := data
CONFIG_DIR := config
EXPERIMENTS_DIR := experiments

# Cores para output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# Configurações de split de dados (customizáveis)
TRAIN_SPLIT := 0.7
VAL_SPLIT := 0.2
TEST_SPLIT := 0.1

# ========================================
# 📋 HELP - Lista todos os comandos
# ========================================

.PHONY: help

help:
	@echo "$(CYAN)🚀 Datalid 3.0 - Makefile Commands$(RESET)"
	@echo "$(CYAN)======================================$(RESET)"
	@echo ""
	@echo "$(GREEN)📦 INSTALAÇÃO:$(RESET)"
	@echo "  install              Instala dependências de produção"
	@echo "  install-dev          Instala dependências de desenvolvimento"
	@echo "  install-all          Instala todas as dependências"
	@echo ""
	@echo "$(GREEN)🔍 VALIDAÇÃO E TESTE:$(RESET)"
	@echo "  test-cuda            Testa disponibilidade CUDA/GPU"
	@echo "  validate-env         Valida ambiente Python"
	@echo "  validate-segment     Valida dataset de SEGMENTAÇÃO"
	@echo "  validate-detect      Valida dataset de DETECÇÃO"
	@echo "  diagnose             Diagnostica labels processados"
	@echo "  test                 Executa testes unitários"
	@echo "  test-cov             Executa testes com cobertura"
	@echo ""
	@echo "$(GREEN)🔄 PROCESSAMENTO DE DADOS:$(RESET)"
	@echo "  process              Processa dados RAW - SEGMENTAÇÃO (INPUT=pasta)"
	@echo "  process-detect       Processa dados RAW - DETECÇÃO (INPUT=pasta)"
	@echo "  quick-process        Processamento rápido (70/20/10) - SEGMENTAÇÃO"
	@echo "  quick-detect         Processamento rápido (70/20/10) - DETECÇÃO"
	@echo ""
	@echo "$(GREEN)📊 CURVA DE APRENDIZADO:$(RESET)"
	@echo "  process-fractions    Cria datasets com frações (25%, 50%, 75%, 100%)"
	@echo "  train-fractions-nano Treina YOLOv8n-seg em todas as frações"
	@echo "  train-fractions-small Treina YOLOv8s-seg em todas as frações"
	@echo "  train-fractions-medium Treina YOLOv8m-seg em todas as frações"
	@echo "  train-all-fractions  Treina TODOS os modelos em todas as frações"
	@echo "  clean-fractions      Remove datasets fracionados"
	@echo ""
	@echo "$(GREEN)🤖 TREINAMENTO YOLO:$(RESET)"
	@echo "  train-nano           Treina YOLOv8n-seg (segmentação - rápido)"
	@echo "  train-small          Treina YOLOv8s-seg (segmentação - recomendado)"
	@echo "  train-medium         Treina YOLOv8m-seg (segmentação - melhor)"
	@echo "  train-detect-nano    Treina YOLOv8n (detecção)"
	@echo "  train-detect-small   Treina YOLOv8s (detecção)"
	@echo "  train-detect-medium  Treina YOLOv8m (detecção)"
	@echo "  train-quick          Teste rápido SEGMENTAÇÃO (10 épocas)"
	@echo "  train-quick-detect   Teste rápido DETECÇÃO (10 épocas)"
	@echo ""
	@echo "$(GREEN)📊 ANÁLISE E VISUALIZAÇÃO:$(RESET)"
	@echo "  tensorboard          Inicia TensorBoard (porta 6006)"
	@echo "  setup-tensorboard    Converte logs YOLO para TensorBoard"
	@echo "  analyze-errors       Analisa erros (MODEL=... DATA=...)"
	@echo "  compare-models       Compara todos os modelos treinados"
	@echo "  list-experiments     Lista todos os experimentos"
	@echo "  list-completed       Lista experimentos concluídos"
	@echo "  generate-report      Gera relatório completo"
	@echo ""
	@echo "$(GREEN)🔮 PREDIÇÃO/INFERÊNCIA:$(RESET)"
	@echo "  predict-image        Predição em uma imagem (MODEL=... IMAGE=...)"
	@echo "  predict-dir          Predição em diretório (MODEL=... DIR=...)"
	@echo "  predict-batch        Predição em lote (MODEL=... IMAGES='...')"
	@echo "  predict-latest       Predição com último modelo (IMAGE=...)"
	@echo ""
	@echo "$(GREEN)🔤 OCR:$(RESET)"
	@echo "  ocr-setup            Instala engines OCR"
	@echo "  ocr-prepare-data     Prepara dataset OCR (DATASET=...)"
	@echo "  ocr-annotate         Interface para anotar ground truth"
	@echo "  ocr-test             Testa um engine (ENGINE=tesseract/easyocr/...)"
	@echo "  ocr-compare          Compara todos os engines OCR"
	@echo "  ocr-benchmark        Benchmark completo (todos os engines)"
	@echo ""
	@echo "$(GREEN)🔗 PIPELINE COMPLETA (YOLO + OCR):$(RESET)"
	@echo "  pipeline-test        Testa pipeline em uma imagem (IMAGE=...)"
	@echo "  pipeline-eval        Avaliação customizada (NUM=X MODE=random/first)"
	@echo "  pipeline-eval-quick  Avaliação rápida (10 imagens)"
	@echo "  pipeline-eval-full   Avaliação completa (todas as imagens)"
	@echo ""
	@echo "$(GREEN)🌐 API REST:$(RESET)"
	@echo "  api-run              Inicia API básica"
	@echo "  api-dev              Inicia API em modo desenvolvimento"
	@echo "  api-start            Inicia API em produção"
	@echo "  api-test             Testa API"
	@echo "  api-health           Health check da API"
	@echo "  api-docs             Mostra URLs da documentação"
	@echo ""

# ========================================
# 📦 INSTALAÇÃO
# ========================================

.PHONY: install install-dev install-all
install:
	@echo "$(GREEN)📦 Instalando dependências de produção...$(RESET)"
	$(PIP) install -r requirements.txt

install-dev:
	@echo "$(GREEN)📦 Instalando dependências de desenvolvimento...$(RESET)"
	$(PIP) install -r requirements-dev.txt

install-all: install install-dev
	@echo "$(GREEN)✅ Todas as dependências instaladas!$(RESET)"

# ========================================
# 🔍 VALIDAÇÃO E TESTE
# ========================================

.PHONY: test-cuda validate-env validate-segment validate-detect diagnose test test-cov

test-cuda:
	@echo "$(YELLOW)🧪 Testando CUDA/GPU...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/setup/test_cuda.py

validate-env:
	@echo "$(YELLOW)🔍 Validando ambiente...$(RESET)"
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
	$(PYTHON) -c "import ultralytics; print('Ultralytics: OK')"
	@echo "$(GREEN)✅ Ambiente validado!$(RESET)"

validate-segment:
	@echo "$(BLUE)✅ Validando dataset de SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py $(DATA_DIR)/processed/v1_segment --detailed

validate-detect:
	@echo "$(BLUE)✅ Validando dataset de DETECÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py $(DATA_DIR)/processed/v1_detect --detailed

diagnose:
	@echo "$(YELLOW)🔍 Diagnosticando labels processados...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/diagnose_labels.py $(DATA_DIR)/processed/v1_segment

test:
	@echo "$(YELLOW)🧪 Executando testes...$(RESET)"
	pytest tests/ -v

test-cov:
	@echo "$(YELLOW)🧪 Executando testes com cobertura...$(RESET)"
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

# ========================================
# 🔄 PROCESSAMENTO DE DADOS
# ========================================

.PHONY: process process-detect quick-process quick-detect

process:
	@echo "$(BLUE)🔄 Processando dados RAW - SEGMENTAÇÃO...$(RESET)"
ifndef INPUT
	@echo "$(RED)❌ Erro: Especifique INPUT=caminho_dos_dados_raw$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--input "$(INPUT)" \
		--output $(DATA_DIR)/processed/ \
		--train-split $(TRAIN_SPLIT) \
		--val-split $(VAL_SPLIT) \
		--test-split $(TEST_SPLIT) \
		--task segment \
		--validate \

process-detect:
	@echo "$(BLUE)🔄 Processando dados RAW - DETECÇÃO...$(RESET)"
ifndef INPUT
	@echo "$(RED)❌ Erro: Especifique INPUT=caminho_dos_dados_raw$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--input "$(INPUT)" \
		--output $(DATA_DIR)/processed/ \
		--train-split $(TRAIN_SPLIT) \
		--val-split $(VAL_SPLIT) \
		--test-split $(TEST_SPLIT) \
		--task detect \
		--validate \

quick-process:
	@echo "$(BLUE)🔄 Processamento rápido (70/20/10) - SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed/ \
		--preset balanced \
		--task segment \
		--validate \

quick-detect:
	@echo "$(BLUE)🔄 Processamento rápido (70/20/10) - DETECÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed/ \
		--preset balanced \
		--task detect \
		--validate

# ========================================
# 📊 PROCESSAMENTO COM FRAÇÕES (LEARNING CURVES)
# ========================================

BASE_DATA := data/processed/v1_segment
FRACTIONS_DIR := data/processed/fractions
FRACTIONS := 0.25 0.50 0.75
FRACTION_CONFIG_DIR := config/yolo/learning_curves
FRACTION_EPOCHS := 100

.PHONY: process-fractions clean-fractions

process-fractions:
	@echo "$(GREEN)📊 Criando datasets com frações dos dados...$(RESET)"
	@echo "$(CYAN)Base: $(BASE_DATA)$(RESET)"
	@echo "$(CYAN)Saída: $(FRACTIONS_DIR)$(RESET)"
	@echo "$(CYAN)Frações: $(FRACTIONS)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_with_fraction.py \
		--base-data $(BASE_DATA) \
		--output-dir $(FRACTIONS_DIR) \
		--fractions $(FRACTIONS) \
		--seed 42

clean-fractions:
	@echo "$(YELLOW)🧹 Removendo datasets fracionados...$(RESET)"
	@if exist "$(FRACTIONS_DIR)" rmdir /s /q "$(FRACTIONS_DIR)"
	@echo "$(GREEN)✅ Datasets fracionados removidos!$(RESET)"

# ========================================
# 🏋️ TREINAMENTO COM FRAÇÕES (LEARNING CURVES)
# ========================================

.PHONY: train-fractions-nano train-fractions-small train-fractions-medium train-all-fractions

train-fractions-nano:
	@echo "$(BLUE)🏋️ Treinando YOLOv8n-seg em todas as frações...$(RESET)"
	@echo "$(CYAN)📊 Fração 1/3: 25%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8n-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_25 \
		--name learning_curve_nano_0.25 \
		--project experiments
	@echo "$(CYAN)📊 Fração 2/3: 50%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8n-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_50 \
		--name learning_curve_nano_0.50 \
		--project experiments
	@echo "$(CYAN)📊 Fração 3/3: 75%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8n-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_75 \
		--name learning_curve_nano_0.75 \
		--project experiments
	@echo "$(GREEN)✅ YOLOv8n-seg treinado em todas as frações!$(RESET)"

train-fractions-small:
	@echo "$(BLUE)🏋️ Treinando YOLOv8s-seg em todas as frações...$(RESET)"
	@echo "$(CYAN)📊 Fração 1/3: 25%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8s-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_25 \
		--name learning_curve_small_0.25 \
		--project experiments
	@echo "$(CYAN)📊 Fração 2/3: 50%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8s-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_50 \
		--name learning_curve_small_0.50 \
		--project experiments
	@echo "$(CYAN)📊 Fração 3/3: 75%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8s-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_75 \
		--name learning_curve_small_0.75 \
		--project experiments
	@echo "$(GREEN)✅ YOLOv8s-seg treinado em todas as frações!$(RESET)"

train-fractions-medium:
	@echo "$(BLUE)🏋️ Treinando YOLOv8m-seg em todas as frações...$(RESET)"
	@echo "$(CYAN)📊 Fração 1/3: 25%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8m-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_25 \
		--name learning_curve_medium_0.25 \
		--project experiments
	@echo "$(CYAN)📊 Fração 2/3: 50%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8m-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_50 \
		--name learning_curve_medium_0.50 \
		--project experiments
	@echo "$(CYAN)📊 Fração 3/3: 75%...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(FRACTION_CONFIG_DIR)/yolov8m-seg-fraction.yaml \
		--data-path $(FRACTIONS_DIR)/fraction_75 \
		--name learning_curve_medium_0.75 \
		--project experiments
	@echo "$(GREEN)✅ YOLOv8m-seg treinado em todas as frações!$(RESET)"

train-all-fractions:
	@echo "$(MAGENTA)🎯 Treinando TODOS os modelos em todas as frações...$(RESET)"
	@echo "$(YELLOW)⚠️ Isso executará 9 treinamentos (pode levar várias horas)$(RESET)"
	make train-fractions-nano
	make train-fractions-small
	make train-fractions-medium
	@echo "$(GREEN)🎉 Todos os modelos treinados!$(RESET)"

# ========================================
# 🤖 TREINAMENTO YOLO
# ========================================

.PHONY: train-nano train-small train-medium train-detect-nano train-detect-small train-detect-medium
.PHONY: train-quick train-quick-detect

# Treinamento de Segmentação
train-nano:
	@echo "$(BLUE)🤖 Treinando YOLOv8n-seg...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/segmentation/yolov8n-seg.yaml \
		--data-path $(DATA_DIR)/processed/v1_segment \
		--name yolov8n-seg \
		--project experiments

train-small:
	@echo "$(BLUE)🤖 Treinando YOLOv8s-seg...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/segmentation/yolov8s-seg.yaml \
		--data-path $(DATA_DIR)/processed/v1_segment \
		--name yolov8s-seg \
		--project experiments

train-medium:
	@echo "$(BLUE)🤖 Treinando YOLOv8m-seg...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/segmentation/yolov8m-seg.yaml \
		--data-path $(DATA_DIR)/processed/v1_segment \
		--name yolov8m-seg \
		--project experiments

# Treinamento de Detecção
train-detect-nano:
	@echo "$(BLUE)🤖 Treinando YOLOv8n (detecção)...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/bbox/yolov8n.yaml \
		--data-path $(DATA_DIR)/processed/v1_detect \
		--name yolov8n-detect \
		--project experiments

train-detect-small:
	@echo "$(BLUE)🤖 Treinando YOLOv8s (detecção)...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/bbox/yolov8s.yaml \
		--data-path $(DATA_DIR)/processed/v1_detect \
		--name yolov8s-detect \
		--project experiments

train-detect-medium:
	@echo "$(BLUE)🤖 Treinando YOLOv8m (detecção)...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/training/train_yolo.py \
		--config $(CONFIG_DIR)/yolo/bbox/yolov8m.yaml \
		--data-path $(DATA_DIR)/processed/v1_detect \
		--name yolov8m-detect \
		--project experiments

# ========================================
# 📊 ANÁLISE E VISUALIZAÇÃO
# ========================================

.PHONY: tensorboard setup-tensorboard analyze-errors compare-models list-experiments list-completed generate-report

setup-tensorboard:
	@echo "$(CYAN)📊 Convertendo logs YOLO para TensorBoard...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/monitoring/setup_tensorboard.py

tensorboard:
	@echo "$(CYAN)📈 Iniciando TensorBoard...$(RESET)"
	@echo "$(YELLOW)💡 Acesse: http://localhost:6006$(RESET)"
	$(PYTHON) -m tensorboard.main --logdir=$(EXPERIMENTS_DIR) --port=6006 --bind_all

analyze-errors:
	@echo "$(CYAN)🔍 Analisando erros...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=path/to/model.pt$(RESET)"
	@exit 1
endif
ifndef DATA
	@echo "$(RED)❌ Erro: Especifique DATA=path/to/dataset$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/analyze_errors.py --model $(MODEL) --data $(DATA)

compare-models:
	@echo "$(CYAN)📊 Comparando modelos...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/compare_models.py --experiments-dir $(EXPERIMENTS_DIR)

test-model:
	@echo "$(CYAN)🧪 Testando modelo YOLO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/test_model.py --model_path="$(EXPERIMENTS_DIR)/$(MODEL)/weights/best.pt"

# ========================================
# 🔮 PREDIÇÃO/INFERÊNCIA
# ========================================

.PHONY: predict-image predict-dir predict-batch predict-latest

predict-image:
	@echo "$(GREEN)🔮 Executando predição em imagem...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@exit 1
endif
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--image $(IMAGE) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}

predict-dir:
	@echo "$(GREEN)🔮 Executando predição em diretório...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@exit 1
endif
ifndef DIR
	@echo "$(RED)❌ Erro: Especifique DIR=caminho/para/diretorio$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--directory $(DIR) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}

predict-batch:
	@echo "$(GREEN)🔮 Executando predição em lote...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@exit 1
endif
ifndef IMAGES
	@echo "$(RED)❌ Erro: Especifique IMAGES='img1.jpg img2.jpg ...'$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--batch $(IMAGES) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}

predict-latest:
	@echo "$(GREEN)🔮 Executando predição com último modelo treinado...$(RESET)"
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_latest.py \
		--image "$(IMAGE)" \
		--conf $(if $(CONF),$(CONF),0.25) \
		--iou $(if $(IOU),$(IOU),0.7) \
		--save-images \
		--save-json

# ========================================
# 🔤 OCR (Optical Character Recognition)
# ========================================

.PHONY: ocr-setup ocr-prepare-data ocr-annotate ocr-test ocr-compare ocr-benchmark

ocr-setup:
	@echo "$(BLUE)🔧 Instalando engines OCR...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/setup/install_ocr_engines.py
	@echo "$(GREEN)✅ OCRs instalados!$(RESET)"

ocr-prepare-data:
	@echo "$(BLUE)📦 Preparando dataset OCR...$(RESET)"
ifndef DATASET
	@echo "$(YELLOW)💡 Usando dataset padrão: data/raw/TCC_DATESET_V2-2$(RESET)"
	$(eval DATASET := data/raw/TCC_DATESET_V2-2)
endif
	$(PYTHON) $(SCRIPTS_DIR)/data/prepare_ocr_dataset.py \
		--dataset $(DATASET) \
		--output $(DATA_DIR)/ocr_test \
		--max-samples 50 \
		--padding 10 \
		$(if $(MASK),--use-mask,) \
		$(if $(MASK_STRATEGY),--mask-strategy $(MASK_STRATEGY),)
	@echo "$(GREEN)✅ Dataset OCR preparado em $(DATA_DIR)/ocr_test$(RESET)"

ocr-annotate:
	@echo "$(BLUE)📝 Iniciando anotação de ground truth...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/annotate_ground_truth.py \
		--data-dir $(DATA_DIR)/ocr_test
	@echo "$(GREEN)✅ Anotação concluída!$(RESET)"

ocr-test:
ifndef ENGINE
	@echo "$(RED)❌ Especifique: make ocr-test ENGINE=paddleocr$(RESET)"
	@echo "$(YELLOW)Engines disponíveis: tesseract, easyocr, openocr, paddleocr, trocr$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)🧪 Testando $(ENGINE)...$(RESET)"
	$(PYTHON) -m src.ocr.evaluator \
		--engine $(ENGINE) \
		--config $(CONFIG_DIR)/ocr/$(ENGINE).yaml \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/ocr_benchmarks/$(ENGINE) \
		$(if $(PREP),--preprocessing $(PREP),)
	@echo "$(GREEN)✅ Teste do $(ENGINE) concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/ocr_benchmarks/$(ENGINE)/$(RESET)"

ocr-compare:
	@echo "$(MAGENTA)📊 Comparando OCRs...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/benchmark_ocrs.py \
		--config $(CONFIG_DIR)/experiments/ocr_comparison.yaml \
		--output outputs/ocr_benchmarks/comparison \
		$(if $(PREP),--preprocessing $(PREP),) \
		$(if $(ENGINE),--engine $(ENGINE))
	@echo "$(GREEN)✅ Comparação concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/ocr_benchmarks/comparison/$(RESET)"

ocr-benchmark:
	@echo "$(MAGENTA)🏆 Benchmark completo de OCRs...$(RESET)"
	@echo "$(YELLOW)⚠️  Isso pode levar alguns minutos...$(RESET)"
	@make ocr-test ENGINE=tesseract
	@make ocr-test ENGINE=easyocr
	@make ocr-test ENGINE=openocr
	@make ocr-test ENGINE=paddleocr
	@make ocr-test ENGINE=trocr
	@make ocr-compare
	@echo "$(GREEN)🎉 Benchmark completo!$(RESET)"

# ========================================
# 🔗 PIPELINE COMPLETA (YOLO + OCR)
# ========================================

.PHONY: pipeline-test pipeline-eval pipeline-eval-quick pipeline-eval-full

pipeline-test:
	@echo "$(MAGENTA)🚀 Testando pipeline completa (YOLO → OCR)...$(RESET)"
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@exit 1
endif
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image "$(IMAGE)" \
		--config config/pipeline/full_pipeline.yaml \
		--output outputs/pipeline_steps
	@echo "$(GREEN)✅ Pipeline testada! Veja os resultados em outputs/pipeline_steps/$(RESET)"

pipeline-eval-full:
	@echo "$(MAGENTA)📊 Avaliação completa da pipeline (todas as imagens)...$(RESET)"
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml
	@echo "$(GREEN)✅ Avaliação concluída! Veja os resultados em outputs/pipeline_evaluation/$(RESET)"

pipeline-eval-quick:
	@echo "$(MAGENTA)📊 Avaliação rápida da pipeline (10 imagens)...$(RESET)"
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml \
		--num-images 10 \
		--selection-mode first
	@echo "$(GREEN)✅ Avaliação rápida concluída!$(RESET)"

pipeline-eval:
	@echo "$(MAGENTA)📊 Avaliação da pipeline...$(RESET)"
ifndef NUM
	@echo "$(RED)❌ Erro: Especifique NUM=número_de_imagens$(RESET)"
	@echo "$(YELLOW)Exemplos:$(RESET)"
	@echo "  make pipeline-eval NUM=10                    # Primeiras 10 imagens"
	@echo "  make pipeline-eval NUM=20 MODE=random        # 20 imagens aleatórias"
	@echo "  make pipeline-eval NUM=50 OUTPUT=meus_testes # Customizar output"
	@exit 1
endif
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml \
		--num-images $(NUM) \
		$(if $(MODE),--selection-mode $(MODE),) \
		$(if $(OUTPUT),--output $(OUTPUT),)
	@echo "$(GREEN)✅ Avaliação concluída!$(RESET)"

# ========================================
# 🌐 API REST
# ========================================

.PHONY: api-run api-dev api-start api-test api-health api-docs

api-run:
	@echo "$(CYAN)🚀 Iniciando API Datalid...$(RESET)"
	$(PYTHON) scripts/api/run_api.py

api-dev:
	@echo "$(CYAN)🔧 Iniciando API em modo desenvolvimento...$(RESET)"
	python scripts/api/start_server.py --dev

api-start:
	@echo "$(GREEN)🚀 Iniciando API em modo produção...$(RESET)"
	python scripts/api/start_server.py --host 0.0.0.0 --port 8000

api-health:
	@echo "$(BLUE)💚 Health check da API...$(RESET)"
	curl http://localhost:8000/health | python -m json.tool

api-docs:
	@echo "$(CYAN)📚 Documentação da API:$(RESET)"
	@echo "  - Swagger UI: http://localhost:8000/docs"
	@echo "  - ReDoc: http://localhost:8000/redoc"
	@echo "  - OpenAPI: http://localhost:8000/openapi.json"