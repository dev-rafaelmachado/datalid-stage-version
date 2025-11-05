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

# FOCO: SEGMENTAÇÃO POLIGONAL (padrão para o projeto)
# DEFAULT_TASKS 


# Cores para output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# Configurações de split de dados (customizáveis)
TRAIN_SPLIT := 0.7
VAL_SPLIT := 0.2
TEST_SPLIT := 0.1

# Configurações do Roboflow
API_KEY := crS7dKMHZj3VlfWw40mS
WORKSPACE := projetotransformadorii
PROJECT := tcc_dateset_v2-zkcsu
VERSION := 2
FORMAT := yolov8

# ========================================
# 📋 HELP - Lista todos os comandos
# ========================================

.PHONY: help help-analysis

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
	@echo "  test-tensorboard     Testa TensorBoard em tempo real ⭐"
	@echo "  validate-env         Valida ambiente Python"
	@echo "  validate-segment     Valida dataset de SEGMENTAÇÃO ⭐"
	@echo "  validate-detect      Valida dataset de DETECÇÃO"
	@echo "  diagnose             Diagnostica labels processados ⭐"
	@echo "  diagnose-raw         Diagnostica labels RAW (INPUT=pasta) ⭐"
	@echo "  test                 Executa testes unitários"
	@echo "  test-cov             Executa testes com cobertura"
	@echo ""
	@echo "$(GREEN)📥 DOWNLOAD DO ROBOFLOW:$(RESET)"
	@echo "  download-dataset     Download básico do dataset"
	@echo "  download-and-process Download + processamento automático"
	@echo "  workflow-complete    Download + processamento + teste"
	@echo ""
	@echo "$(GREEN)🔄 PROCESSAMENTO DE DADOS (FOCO: SEGMENTAÇÃO ⭐):$(RESET)"
	@echo "  process              Processa dados RAW (INPUT=pasta) - SEGMENTAÇÃO ⭐"
	@echo "  process-segment      Alias para process - SEGMENTAÇÃO ⭐"
	@echo "  process-detect       Processa dados RAW - Apenas Detecção (bbox)"
	@echo "  process-both         Processa dados RAW - Segmentação + Detecção"
	@echo "  validate-dataset     Valida dataset YOLO (interactive)"
	@echo "  quick-process        Processamento rápido (70/20/10) - SEGMENTAÇÃO ⭐"
	@echo "  quick-detect         Processamento rápido - Detecção"
	@echo "  research-process     Processamento para pesquisa (80/10/10) - SEGMENTAÇÃO ⭐"
	@echo ""
	@echo "$(GREEN)🤖 TREINAMENTO (FOCO: SEGMENTAÇÃO POLIGONAL):$(RESET)"
	@echo "  train-nano           Treina YOLOv8n-seg ⭐ (segmentação - rápido)"
	@echo "  train-small          Treina YOLOv8s-seg ⭐ (segmentação - recomendado)"
	@echo "  train-medium         Treina YOLOv8m-seg ⭐ (segmentação - melhor)"
	@echo "  train-detect-nano    Treina YOLOv8n (bbox apenas)"
	@echo "  train-detect-small   Treina YOLOv8s (bbox apenas)"
	@echo "  train-detect-medium  Treina YOLOv8m (bbox apenas)"
	@echo ""
	@echo "$(GREEN)🎛️ TREINAMENTO $(RESET)"
	@echo "  train-quick          Teste rápido SEGMENTAÇÃO (10 épocas) ⭐"
	@echo "  train-quick-detect   Teste rápido Detecção (10 épocas)"
	@echo "  train-dev            Desenvolvimento SEGMENTAÇÃO ⭐"
	@echo "  train-dev-detect     Desenvolvimento Detecção"
	@echo "  train-final-nano     FINAL TCC - YOLOv8n-seg ⭐"
	@echo "  train-final-small    FINAL TCC - YOLOv8s-seg ⭐"
	@echo "  train-final-medium   FINAL TCC - YOLOv8m-seg ⭐"
	@echo "  train-compare-all    Treina modelos segmentação (comparação) ⭐"
	@echo "  train-compare-detect Treina modelos detecção (comparação)"
	@echo "  train-overnight      Treinamento overnight segmentação (200 épocas) ⭐"
	@echo ""
	@echo "$(GREEN)📊 CURVA DE APRENDIZADO (LEARNING CURVES):$(RESET)"
	@echo "  process-fractions    Cria datasets com frações (25%, 50%, 75%, 100%) ⭐"
	@echo "  train-fractions-nano Treina YOLOv8n-seg em todas as frações ⭐"
	@echo "  train-fractions-small Treina YOLOv8s-seg em todas as frações ⭐"
	@echo "  train-fractions-medium Treina YOLOv8m-seg em todas as frações ⭐"
	@echo "  train-all-fractions  Treina TODOS os modelos em todas as frações ⭐"
	@echo "  compare-learning-curves Analisa e compara curvas de aprendizado ⭐"
	@echo ""
	@echo "$(GREEN)� OCR (OPTICAL CHARACTER RECOGNITION):$(RESET)"
	@echo "  ocr-setup            Instala engines OCR ⭐"
	@echo "  ocr-prepare-data     Prepara dataset OCR a partir de detecções YOLO ⭐"
	@echo "  ocr-annotate         Interface para anotar ground truth ⭐"
	@echo "  ocr-test             Testa um engine específico (ENGINE=paddleocr PREP=medium) ⭐"
	@echo "  ocr-compare          Compara engines (ENGINE=tesseract PREP=heavy) ⭐"
	@echo "  ocr-benchmark        Benchmark completo de todos os engines"
	@echo ""
	@echo "$(CYAN)  PARSeq - Escolha o modelo ideal:$(RESET)"
	@echo "  ocr-parseq           Testa PARSeq BASE (melhor multi-linha) ✅ RECOMENDADO"
	@echo "  ocr-parseq-tiny      Testa PARSeq TINY (rápido, ⚠️ ruim multi-linha)"
	@echo "  ocr-parseq-base      Testa PARSeq BASE (melhor multi-linha) ⭐"
	@echo "  ocr-parseq-large     Testa PARSeq LARGE (máxima precisão) 🏆"
	@echo "  ocr-parseq-compare   Compara TODOS os modelos PARSeq 📊"
	@echo "  ocr-parseq-analyze   Analisa resultados (sem rodar testes) 📈"
	@echo "  ocr-parseq-setup     Configura e baixa TODOS os modelos PARSeq"
	@echo "  ocr-parseq-validate  Valida implementação completa do PARSeq"
	@echo ""
	@echo "$(MAGENTA)  🚀 Enhanced PARSeq - Pipeline Robusto (NOVO):$(RESET)"
	@echo "  ocr-enhanced-demo             Demo interativo (IMAGE=test.jpg) ⭐⭐⭐"
	@echo "  ocr-enhanced                  Teste completo (balanceado) ⭐"
	@echo "  ocr-enhanced-fast             Modo rápido (sem ensemble)"
	@echo "  ocr-enhanced-quality          Modo alta qualidade (lento)"
	@echo "  ocr-enhanced-batch            Processar diretório (DIR=...) 📦"
	@echo "  ocr-enhanced-ablation         Estudo de ablação 🔬"
	@echo "  ocr-enhanced-vs-baseline      Comparar vs baseline 📊"
	@echo "  ocr-enhanced-finetune         Fine-tuning 🎓"
	@echo "  workflow-enhanced-parseq      Workflow completo 🎯"
	@echo "  help-enhanced-parseq          Ajuda detalhada"
	@echo ""
	@echo "$(CYAN)  Outros comandos OCR:$(RESET)"
	@echo "  ocr-trocr            Testa TrOCR (microsoft/trocr-base-printed) ✅ COM normalização"
	@echo "  ocr-trocr-quick      Teste rápido TrOCR (10 imagens) ⚡"
	@echo "  ocr-trocr-benchmark  Benchmark completo do TrOCR 🏆"
	@echo "  ocr-trocr-validate-brightness  Valida normalização de brilho 🔆"
	@echo ""
	@echo "$(GREEN)🔗 PIPELINE COMPLETA (YOLO + OCR):$(RESET)"
	@echo "  pipeline-test        Testa pipeline em uma imagem (IMAGE=...) ⚡"
	@echo "  pipeline-eval-quick  Avaliação rápida (10 imagens) 🚀"
	@echo "  pipeline-eval-full   Avaliação completa (todas as imagens) 📊"
	@echo "  pipeline-eval        Avaliação customizada (NUM=X MODE=random/first) 🎯"
	@echo ""
	@echo "$(YELLOW)  Exemplos de uso da pipeline:$(RESET)"
	@echo "    make pipeline-test IMAGE=data/sample.jpg"
	@echo "    make pipeline-eval NUM=10 MODE=first"
	@echo "    make pipeline-eval NUM=20 MODE=random OUTPUT=meus_testes"
	@echo "    make pipeline-eval-full"
	@echo ""
	@echo "$(CYAN)  📖 Documentação da Pipeline:$(RESET)"
	@echo "    docs/PIPELINE_EVALUATION_TLDR.md     - Resumo rápido (2 min)"
	@echo "    docs/PIPELINE_EVALUATION_QUICK.md    - Guia rápido (5 min)"
	@echo "    docs/PIPELINE_EVALUATION_EXAMPLE.md  - Tutorial completo"
	@echo "    docs/PIPELINE_EVALUATION_INDEX.md    - Índice completo"
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

.PHONY: test-cuda validate-env test test-cov test-tensorboard
test-cuda:
	@echo "$(YELLOW)🧪 Testando CUDA/GPU...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/setup/test_cuda.py

validate-env:
	@echo "$(YELLOW)🔍 Validando ambiente...$(RESET)"
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
	$(PYTHON) -c "import ultralytics; print('Ultralytics: OK')"
	@echo "$(GREEN)✅ Ambiente validado!$(RESET)"

test:
	@echo "$(YELLOW)🧪 Executando testes...$(RESET)"
	pytest tests/ -v

test-cov:
	@echo "$(YELLOW)🧪 Executando testes com cobertura...$(RESET)"
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

# Validação de datasets específicos
.PHONY: validate-segment validate-detect

validate-segment:
	@echo "$(BLUE)✅ Validando dataset de SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py $(DATA_DIR)/processed/v1_segment --detailed

validate-detect:
	@echo "$(BLUE)✅ Validando dataset de DETECÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py $(DATA_DIR)/processed/v1_detect --detailed

# Diagnóstico de labels (para identificar problemas)
.PHONY: diagnose diagnose-raw

diagnose:
	@echo "$(YELLOW)🔍 Diagnosticando labels processados...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/diagnose_labels.py $(DATA_DIR)/processed/v1_segment

diagnose-raw:
	@echo "$(YELLOW)🔍 Diagnosticando labels RAW...$(RESET)"
ifndef INPUT
	@echo "$(RED)❌ Erro: Especifique INPUT=caminho_dos_dados_raw$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/data/diagnose_labels.py "$(INPUT)"

# ========================================
# 🔄 PROCESSAMENTO DE DADOS
# ========================================

.PHONY: validate-dataset quick-process research-process process-data process-data-auto

validate-dataset:
	@echo "$(BLUE)✅ Validação interativa de dataset...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py --help
	@echo ""
	@echo "$(CYAN)Exemplo de uso:$(RESET)"
	@echo "$(PYTHON) $(SCRIPTS_DIR)/data/validate_dataset.py data/processed/v1_segment --detailed"

quick-process:
	@echo "$(BLUE)🔄 Processamento rápido (70/20/10) - SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--raw-path $(DATA_DIR)/raw \
		--output-path $(DATA_DIR)/processed/v1_segment \
		--preset balanced \
		--task-type segment \
		--validate-raw \
		--validate-output

quick-detect:
	@echo "$(BLUE)🔄 Processamento rápido (70/20/10) - DETECÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--raw-path $(DATA_DIR)/raw \
		--output-path $(DATA_DIR)/processed/v1_detect \
		--preset balanced \
		--task-type detect \
		--validate-raw \
		--validate-output

# Alias para compatibilidade
quick-process-detect: quick-detect

research-process:
	@echo "$(BLUE)🔄 Processamento para pesquisa (80/10/10) - SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/data/process_raw_data.py \
		--raw-path $(DATA_DIR)/raw \
		--output-path $(DATA_DIR)/processed/v1_segment \
		--preset research \
		--task-type segment \
		--validate-raw \
		--validate-output

# ========================================
# 📊 PROCESSAMENTO COM FRAÇÕES (LEARNING CURVES)
# ========================================

# Configurações padrão para frações
BASE_DATA := data/processed/v1_segment
FRACTIONS_DIR := data/processed/fractions
FRACTIONS := 0.25 0.50 0.75

# Configurações para treinamento com frações
FRACTION_CONFIG_DIR := config/yolo/learning_curves
FRACTION_EPOCHS := 100

.PHONY: process-fractions clean-fractions
process-fractions:
	@echo "$(GREEN)📊 Criando datasets com frações dos dados...$(RESET)"
	@echo "$(CYAN)Base: $(BASE_DATA)$(RESET)"
	@echo "$(CYAN)Saída: $(FRACTIONS_DIR)$(RESET)"
	@echo "$(CYAN)Frações: $(FRACTIONS)$(RESET)"
	@echo ""
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

# Treinar YOLOv8n-seg em todas as frações
.PHONY: train-fractions-nano
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

# Treinar YOLOv8s-seg em todas as frações
.PHONY: train-fractions-small
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

# Treinar YOLOv8m-seg em todas as frações
.PHONY: train-fractions-medium
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

# Treinar TODOS os modelos em todas as frações
.PHONY: train-all-fractions
train-all-fractions:
	@echo "$(MAGENTA)🎯 Treinando TODOS os modelos em todas as frações...$(RESET)"
	@echo "$(YELLOW)⚠️ Isso executará 9 treinamentos (pode levar várias horas)$(RESET)"
	@echo ""
	make train-fractions-nano
	make train-fractions-small
	make train-fractions-medium
	@echo "$(GREEN)🎉 Todos os modelos treinados!$(RESET)"

# ========================================
# 📊 ANÁLISE
# ========================================

.PHONY: tensorboard setup-tensorboard compare-models analyze-errors analyze-best-model compare-segments

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
	@echo "$(RED)❌ Erro: Especifique o modelo com MODEL=path/to/model.pt$(RESET)"
	@exit 1
endif
ifndef DATA
	@echo "$(RED)❌ Erro: Especifique o dataset com DATA=path/to/dataset$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/analyze_errors.py --model $(MODEL) --data $(DATA)

compare-models:
	@echo "$(CYAN)📊 Comparando modelos...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/compare_models.py --experiments-dir $(EXPERIMENTS_DIR)

# Atalho para analisar o melhor modelo de segmentação
analyze-best-model:
	@echo "$(CYAN)🔍 Analisando melhor modelo de segmentação...$(RESET)"
	@latest_model=$$(ls -t $(EXPERIMENTS_DIR)/*/weights/best.pt 2>/dev/null | head -1); \
	if [ -z "$$latest_model" ]; then \
		echo "$(RED)❌ Nenhum modelo encontrado em $(EXPERIMENTS_DIR)$(RESET)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)Analisando: $$latest_model$(RESET)"; \
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/analyze_best_model.py --model "$$latest_model" --data $(DATA_DIR)/processed/v1_segment

# Comparar apenas modelos de segmentação
compare-segments:
	@echo "$(CYAN)📊 Comparando modelos de SEGMENTAÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/compare_models.py --experiments-dir $(EXPERIMENTS_DIR) --pattern "*-seg-*"

# Comparar apenas modelos de detecção
compare-detects:
	@echo "$(CYAN)📊 Comparando modelos de DETECÇÃO...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/evaluation/compare_models.py --experiments-dir $(EXPERIMENTS_DIR) --pattern "*-detect-*"

# ========================================
# 📊 PREDIÇÃO/INFERÊNCIA
# ========================================

.PHONY: predict predict-image predict-dir predict-batch predict-latest
predict:
	@echo "$(MAGENTA)🔮 Predição com modelo YOLO$(RESET)"
	@echo "$(CYAN)Use os comandos específicos abaixo:$(RESET)"
	@echo "  predict-image       Predição em uma imagem"
	@echo "  predict-dir         Predição em diretório"
	@echo "  predict-batch       Predição em lote"
	@echo "  predict-latest      Predição com último modelo treinado"

# Predição em uma única imagem
predict-image:
	@echo "$(GREEN)🔮 Executando predição em imagem...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-image MODEL=experiments/yolov8s-seg_final/weights/best.pt IMAGE=test.jpg$(RESET)"
	@exit 1
endif
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-image MODEL=experiments/yolov8s-seg_final/weights/best.pt IMAGE=test.jpg$(RESET)"
	@exit 1
endif
	@echo "$(CYAN)📸 Modelo: $(MODEL)$(RESET)"
	@echo "$(CYAN)🖼️ Imagem: $(IMAGE)$(RESET)"
	@echo "$(CYAN)💾 Salvando em: outputs/predictions/$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--image $(IMAGE) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}
	@echo "$(GREEN)✅ Predição concluída!$(RESET)"
	@echo "$(YELLOW)📁 Resultados salvos em: outputs/predictions/"

# Predição em diretório
predict-dir:
	@echo "$(GREEN)🔮 Executando predição em diretório...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-dir MODEL=best.pt DIR=data/test/$(RESET)"
	@exit 1
endif
ifndef DIR
	@echo "$(RED)❌ Erro: Especifique DIR=caminho/para/diretorio$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-dir MODEL=best.pt DIR=data/test/$(RESET)"
	@exit 1
endif
	@echo "$(CYAN)📸 Modelo: $(MODEL)$(RESET)"
	@echo "$(CYAN)📁 Diretório: $(DIR)$(RESET)"
	@echo "$(CYAN)💾 Salvando em: outputs/predictions/$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--directory $(DIR) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}
	@echo "$(GREEN)✅ Predição concluída!$(RESET)"
	@echo "$(YELLOW)📁 Resultados salvos em: outputs/predictions/"

# Predição em lote (lista de imagens)
predict-batch:
	@echo "$(GREEN)🔮 Executando predição em lote...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-batch MODEL=best.pt IMAGES='img1.jpg img2.jpg img3.jpg'$(RESET)"
	@exit 1
endif
ifndef IMAGES
	@echo "$(RED)❌ Erro: Especifique IMAGES='img1.jpg img2.jpg ...'$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-batch MODEL=best.pt IMAGES='img1.jpg img2.jpg img3.jpg'$(RESET)"
	@exit 1
endif
	@echo "$(CYAN)📸 Modelo: $(MODEL)$(RESET)"
	@echo "$(CYAN)🖼️ Imagens: $(IMAGES)$(RESET)"
	@echo "$(CYAN)💾 Salvando em: outputs/predictions/$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_yolo.py \
		--model $(MODEL) \
		--batch $(IMAGES) \
		--output-dir outputs/predictions \
		--save-images \
		--save-json \
		--conf $${CONF:-0.25} \
		--iou $${IOU:-0.7}
	@echo "$(GREEN)✅ Predição concluída!$(RESET)"
	@echo "$(YELLOW)📁 Resultados salvos em: outputs/predictions/"

# Predição com último modelo treinado (automático)
predict-latest:
	@echo "$(GREEN)🔮 Executando predição com último modelo treinado...$(RESET)"
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@echo "$(YELLOW)Exemplo: make predict-latest IMAGE=test.jpg$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/predict_latest.py \
		--image "$(IMAGE)" \
		--conf $(if $(CONF),$(CONF),0.25) \
		--iou $(if $(IOU),$(IOU),0.7) \
		--save-images \
		--save-json

# Teste rápido de inferência (modelo + imagem customizáveis)
test-inference:
	@echo "$(GREEN)🧪 Teste de inferência...$(RESET)"
ifndef MODEL
	@echo "$(RED)❌ Erro: Especifique MODEL=caminho/para/weights.pt$(RESET)"
	@echo "$(YELLOW)Exemplo: make test-inference MODEL=experiments/yolov8s-seg_final/weights/best.pt IMAGE=test.jpg$(RESET)"
	@exit 1
endif
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@echo "$(YELLOW)Exemplo: make test-inference MODEL=experiments/yolov8s-seg_final/weights/best.pt IMAGE=test.jpg$(RESET)"
	@exit 1
endif
	$(PYTHON) $(SCRIPTS_DIR)/inference/test_inference.py \
		--model "$(MODEL)" \
		--image "$(IMAGE)" \
		--conf $(if $(CONF),$(CONF),0.25) \
		--iou $(if $(IOU),$(IOU),0.7) \
		$(if $(CROPS),--save-crops,)

# ========================================
# 🔗 PIPELINE COMPLETA (YOLO + OCR)
# ========================================

.PHONY: pipeline-test pipeline-eval pipeline-eval-quick pipeline-eval-full evaluate-pipeline evaluate-pipeline-quick evaluate-pipeline-full

# Teste rápido da pipeline em uma imagem
pipeline-test:
	@echo "$(MAGENTA)🚀 Testando pipeline completa (YOLO → OCR)...$(RESET)"
ifndef IMAGE
	@echo "$(RED)❌ Erro: Especifique IMAGE=caminho/para/imagem.jpg$(RESET)"
	@echo "$(YELLOW)Exemplo: make pipeline-test IMAGE=data/sample.jpg$(RESET)"
	@exit 1
endif
	@echo "$(CYAN)📸 Imagem: $(IMAGE)$(RESET)"
	@echo "$(CYAN)💾 Salvando em: outputs/pipeline_steps/$(RESET)"
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image "$(IMAGE)" \
		--config config/pipeline/full_pipeline.yaml \
		--output outputs/pipeline_steps
	@echo "$(GREEN)✅ Pipeline testada! Veja os resultados em outputs/pipeline_steps/$(RESET)"

# Avaliação completa da pipeline (todas as imagens)
pipeline-eval-full:
	@echo "$(MAGENTA)📊 Avaliação completa da pipeline (todas as imagens)...$(RESET)"
	@echo "$(CYAN)Dataset: data/ocr_test/$(RESET)"
	@echo "$(CYAN)Configuração: config/pipeline/pipeline_evaluation.yaml$(RESET)"
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml
	@echo "$(GREEN)✅ Avaliação concluída! Veja os resultados em outputs/pipeline_evaluation/$(RESET)"

# Avaliação rápida (10 imagens)
pipeline-eval-quick:
	@echo "$(MAGENTA)📊 Avaliação rápida da pipeline (10 imagens)...$(RESET)"
	@echo "$(CYAN)Seleção: primeiras 10 imagens$(RESET)"
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml \
		--num-images 10 \
		--selection-mode first
	@echo "$(GREEN)✅ Avaliação rápida concluída!$(RESET)"

# Avaliação customizada
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
	@echo "$(CYAN)Imagens: $(NUM)$(RESET)"
	@echo "$(CYAN)Modo: $(if $(MODE),$(MODE),first)$(RESET)"
	$(PYTHON) scripts/pipeline/evaluate_pipeline.py \
		--config config/pipeline/pipeline_evaluation.yaml \
		--num-images $(NUM) \
		$(if $(MODE),--selection-mode $(MODE),) \
		$(if $(OUTPUT),--output $(OUTPUT),)
	@echo "$(GREEN)✅ Avaliação concluída!$(RESET)"

# Aliases for pipeline evaluation (alternative names)
evaluate-pipeline: pipeline-eval
evaluate-pipeline-quick: pipeline-eval-quick
evaluate-pipeline-full: pipeline-eval-full

# ========================================
# 🚀 API E DEPLOY
# ========================================

.PHONY: run-api build-docker run-docker
run-api:
	@echo "$(GREEN)🌐 Iniciando API de desenvolvimento...$(RESET)"
	@echo "$(YELLOW)💡 Acesse: http://localhost:8000$(RESET)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

build-docker:
	@echo "$(BLUE)🐳 Construindo imagem Docker...$(RESET)"
	docker build -t $(PROJECT_NAME):$(VERSION) .

run-docker:
	@echo "$(BLUE)🐳 Executando container Docker...$(RESET)"
	docker run -p 8000:8000 $(PROJECT_NAME):$(VERSION)

# ========================================
# 🧹 LIMPEZA
# ========================================

.PHONY: clean clean-data clean-models clean-all
clean:
	@echo "$(RED)🧹 Limpando arquivos temporários...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

clean-data:
	@echo "$(RED)🧹 Removendo dados processados...$(RESET)"
	rm -rf $(DATA_DIR)/processed/*
	@echo "$(YELLOW)⚠️ Dados RAW mantidos em $(DATA_DIR)/raw$(RESET)"

clean-models:
	@echo "$(RED)🧹 Removendo modelos treinados...$(RESET)"
	rm -rf $(EXPERIMENTS_DIR)/*
	rm -rf runs/

clean-all: clean clean-data clean-models
	@echo "$(RED)🧹 Limpeza completa realizada!$(RESET)"

# Limpeza específica para experimentos de learning curves
.PHONY: clean-learning-curves

clean-learning-curves:
	@echo "$(YELLOW)🧹 Removendo experimentos de learning curves...$(RESET)"
	@if exist "$(EXPERIMENTS_DIR)\learning_curve_*" rmdir /s /q "$(EXPERIMENTS_DIR)\learning_curve_*"
	@if exist "outputs\learning_curves" rmdir /s /q "outputs\learning_curves"
	@echo "$(GREEN)✅ Experimentos de learning curves removidos!$(RESET)"

# ========================================
# 🎯 COMANDOS DE CONVENIÊNCIA
# ========================================

.PHONY: setup quick-start full-pipeline

setup: install-all test-cuda validate-env
	@echo "$(GREEN)🎉 Setup completo! Sistema pronto para uso.$(RESET)"
	@echo "$(CYAN)📋 Próximos passos sugeridos (SEGMENTAÇÃO):$(RESET)"
	@echo "  1. make process INPUT=data/raw/dataset  # Processar dados"
	@echo "  2. make train-quick                      # Teste rápido"
	@echo "  3. make train-final-small                # Treinamento final"

quick-start: setup quick-process train-quick
	@echo "$(GREEN)🚀 Quick start completo - SEGMENTAÇÃO POLIGONAL!$(RESET)"
	@echo "$(CYAN)Próximos passos:$(RESET)"
	@echo "  1. make tensorboard      # Ver métricas"
	@echo "  2. make validate-segment # Validar dataset"
	@echo "  3. make train-final-small # Treinamento final"

quick-start-detect: setup quick-detect train-detect-small
	@echo "$(GREEN)🚀 Quick start completo - DETECÇÃO (bbox)!$(RESET)"
	@echo "$(CYAN)Próximos passos:$(RESET)"
	@echo "  1. make tensorboard     # Ver métricas"
	@echo "  2. make validate-detect # Validar dataset"

full-pipeline: setup research-process train-nano train-small train-medium
	@echo "$(GREEN)🎯 Pipeline completo executado - SEGMENTAÇÃO POLIGONAL!$(RESET)"
	@echo "$(CYAN)Resultados em: $(EXPERIMENTS_DIR)$(RESET)"
	@echo "$(YELLOW)📊 Use 'make compare-final' para comparar modelos$(RESET)"

full-pipeline-detect: setup quick-detect train-detect-nano train-detect-small train-detect-medium
	@echo "$(GREEN)🎯 Pipeline completo executado - DETECÇÃO (bbox)!$(RESET)"
	@echo "$(CYAN)Resultados em: $(EXPERIMENTS_DIR)$(RESET)"

# ========================================
# 📊 WORKFLOW DE LEARNING CURVES
# ========================================

# Workflow completo para análise de curvas de aprendizado
.PHONY: workflow-learning-curves workflow-learning-curves-quick

workflow-learning-curves:
	@echo "$(MAGENTA)📊 WORKFLOW COMPLETO - ANÁLISE DE CURVAS DE APRENDIZADO$(RESET)"
	@echo "$(CYAN)Este workflow analisa o aprendizado dos modelos com diferentes frações de dados$(RESET)"
	@echo ""
	@echo "$(BLUE)1/4 📊 Criando datasets fracionados (25%%, 50%%, 75%%, 100%%)...$(RESET)"
	make process-fractions
	@echo ""
	@echo "$(BLUE)2/4 🤖 Treinando TODOS os modelos em todas as frações...$(RESET)"
	@echo "$(YELLOW)⚠️  ATENÇÃO: Este processo pode levar MUITO tempo!$(RESET)"
	make train-all-fractions
	@echo ""
	@echo "$(BLUE)3/4 📈 Analisando e comparando resultados...$(RESET)"
	make compare-learning-curves
	@echo ""
	@echo "$(BLUE)4/4 ✅ Workflow concluído!$(RESET)"
	@echo "$(GREEN)🎉 ANÁLISE DE CURVAS DE APRENDIZADO CONCLUÍDA!$(RESET)"
	@echo "$(YELLOW)📊 Datasets fracionados: $(FRACTIONS_DIR)$(RESET)"
	@echo "$(YELLOW)🤖 Experimentos: $(EXPERIMENTS_DIR)/learning_curve_*$(RESET)"
	@echo "$(YELLOW)📈 Resultados: outputs/learning_curves/$(RESET)"

# Workflow rápido (apenas nano model)
workflow-learning-curves-quick:
	@echo "$(MAGENTA)📊 WORKFLOW RÁPIDO - LEARNING CURVES (apenas YOLOv8n-seg)$(RESET)"
	@echo ""
	@echo "$(BLUE)1/3 📊 Criando datasets fracionados...$(RESET)"
	make process-fractions
	@echo ""
	@echo "$(BLUE)2/3 🤖 Treinando YOLOv8n-seg em todas as frações...$(RESET)"
	make train-fractions-nano
	@echo ""
	@echo "$(BLUE)3/3 📈 Analisando resultados...$(RESET)"
	make compare-learning-curves
	@echo ""
	@echo "$(GREEN)🎉 ANÁLISE RÁPIDA CONCLUÍDA!$(RESET)"
	@echo "$(YELLOW)📈 Resultados: outputs/learning_curves/$(RESET)"

# ========================================
# 📝 INFORMAÇÕES
# ========================================

.PHONY: info status version
info:
	@echo "$(CYAN)📋 Informações do Projeto$(RESET)"
	@echo "$(CYAN)========================$(RESET)"
	@echo "Nome: $(PROJECT_NAME)"
	@echo "Versão: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Diretório: $(shell pwd)"

status:
	@echo "$(CYAN)📊 Status do Sistema$(RESET)"
	@echo "$(CYAN)==================$(RESET)"
	@echo "Dados RAW: $(shell find $(DATA_DIR)/raw -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l) imagens"
	@echo "Dados processados: $(shell find $(DATA_DIR)/processed -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l) imagens"
	@echo "Experimentos: $(shell find $(EXPERIMENTS_DIR) -maxdepth 1 -type d 2>/dev/null | wc -l) runs"

version:
	@echo "$(PROJECT_NAME) v$(VERSION)"

# ========================================
# 🎛️ COMANDOS DO NOVO SISTEMA
# ========================================

.PHONY: list-experiments list-completed compare-final generate-report cleanup-failed
list-experiments:
	@echo "$(BLUE)📊 Listando experimentos...$(RESET)"
	python $(SCRIPTS_DIR)/experiments/manage_experiments.py list

list-completed:
	@echo "$(GREEN)✅ Experimentos concluídos...$(RESET)"
	python $(SCRIPTS_DIR)/experiments/manage_experiments.py list --status completed --sort map50

compare-final:
	@echo "$(YELLOW)📈 Comparando experimentos finais...$(RESET)"
	python $(SCRIPTS_DIR)/experiments/manage_experiments.py compare \
		final_yolov8n_detect final_yolov8s_detect final_yolov8m_detect \
		--output experiments/final_comparison.png

generate-report:
	@echo "$(MAGENTA)📝 Gerando relatório completo...$(RESET)"
	python $(SCRIPTS_DIR)/experiments/manage_experiments.py report --output experiments/relatorio_completo.md

cleanup-failed:
	@echo "$(RED)🗑️ Limpando experimentos falhados...$(RESET)"
	python $(SCRIPTS_DIR)/experiments/manage_experiments.py cleanup --dry-run


# ========================================
# 🔤 OCR (Optical Character Recognition)
# ========================================

.PHONY: ocr-setup ocr-prepare-data ocr-test ocr-compare ocr-benchmark
.PHONY: ocr-trocr ocr-trocr-quick ocr-trocr-benchmark ocr-trocr-validate-brightness
.PHONY: prep-test prep-compare pipeline-test pipeline-run
.PHONY: exp-ocr-comparison viz-ocr-results viz-preprocessing
.PHONY: ocr-test-stats

# Teste do Sistema de Estatísticas
ocr-test-stats:
	@echo "$(BLUE)🧪 Testando sistema de estatísticas OCR...$(RESET)"
	@echo "$(YELLOW)Gerando dados mock e visualizações...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/utils/test_ocr_statistics.py
	@echo "$(GREEN)✅ Teste concluído!$(RESET)"
	@echo "$(CYAN)💡 Ver resultados em: outputs/test_statistics/report.html$(RESET)"

# Setup e Instalação
ocr-setup:
	@echo "$(BLUE)🔧 Instalando engines OCR...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/setup/install_ocr_engines.py
	@echo "$(GREEN)✅ OCRs instalados!$(RESET)"

# Teste rápido do módulo
ocr-test-module:
	@echo "$(BLUE)🧪 Testando módulo OCR...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/test_ocr_module.py

# Preparação de Dados
ocr-prepare-data:
	@echo "$(BLUE)📦 Preparando dataset OCR...$(RESET)"
ifndef DATASET
	@echo "$(YELLOW)💡 Usando dataset padrão: data/raw/TCC_DATESET_V2-2$(RESET)"
	$(eval DATASET := data/raw/TCC_DATESET_V2-2)
endif
	@if not exist "$(DATASET)" ( \
		echo "$(RED)❌ Dataset não encontrado: $(DATASET)$(RESET)" && \
		echo "$(YELLOW)💡 Certifique-se que o dataset está baixado$(RESET)" && \
		exit 1 \
	)
	$(PYTHON) $(SCRIPTS_DIR)/data/prepare_ocr_dataset.py \
		--dataset $(DATASET) \
		--output $(DATA_DIR)/ocr_test \
		--max-samples 50 \
		--padding 10 \
		$(if $(MASK),--use-mask,) \
		$(if $(MASK_STRATEGY),--mask-strategy $(MASK_STRATEGY),)
	@echo "$(GREEN)✅ Dataset OCR preparado em $(DATA_DIR)/ocr_test$(RESET)"
	@echo "$(CYAN)💡 Use MASK=1 para aplicar máscaras de segmentação$(RESET)"
	@echo "$(CYAN)💡 Use MASK_STRATEGY=white/black/blur para ajustar background$(RESET)"

# Anotação de Ground Truth
ocr-annotate:
	@echo "$(BLUE)📝 Iniciando anotação de ground truth...$(RESET)"
	@if not exist "$(DATA_DIR)\ocr_test\images" ( \
		echo "$(RED)❌ Dataset OCR não encontrado!$(RESET)" && \
		echo "$(YELLOW)💡 Execute primeiro: make ocr-prepare-data$(RESET)" && \
		exit 1 \
	)
	$(PYTHON) $(SCRIPTS_DIR)/data/annotate_ground_truth.py \
		--data-dir $(DATA_DIR)/ocr_test
	@echo "$(GREEN)✅ Anotação concluída!$(RESET)"

# Teste Individual de Engine
ocr-test:
ifndef ENGINE
	@echo "$(RED)❌ Especifique: make ocr-test ENGINE=paddleocr$(RESET)"
	@echo "$(YELLOW)Engines disponíveis: tesseract, easyocr, openocr, paddleocr, trocr, parseq$(RESET)"
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
	@echo ""
	@echo "$(CYAN)� RESULTADOS GERADOS:$(RESET)"
	@echo "$(CYAN)  📄 HTML Report: outputs/ocr_benchmarks/$(ENGINE)/report.html$(RESET)"
	@echo "$(CYAN)  📝 Markdown Report: outputs/ocr_benchmarks/$(ENGINE)/report.md$(RESET)"
	@echo "$(CYAN)  📈 Estatísticas JSON: outputs/ocr_benchmarks/$(ENGINE)/statistics.json$(RESET)"
	@echo "$(CYAN)  📊 Visualizações: outputs/ocr_benchmarks/$(ENGINE)/*.png$(RESET)"
	@echo ""
	@echo "$(MAGENTA)💡 Gráficos gerados:$(RESET)"
	@echo "   - overview.png - Visão geral de todas as métricas"
	@echo "   - error_distribution.png - Análise de distribuição de erros"
	@echo "   - confidence_analysis.png - Confiança vs acurácia"
	@echo "   - length_analysis.png - Impacto do comprimento do texto"
	@echo "   - time_analysis.png - Análise de tempo de processamento"
	@echo "   - character_confusion.png - Matriz de confusão de caracteres"
	@echo "   - performance_summary.png - Dashboard de performance"
	@echo "   - error_examples.png - Exemplos de casos com alto erro"
	@echo ""
	@echo "$(CYAN)💡 Cada OCR usa seu pré-processamento otimizado automaticamente$(RESET)"
	@echo "$(CYAN)💡 Use PREP=ppro-none para desabilitar ou PREP=ppro-{engine} para especificar$(RESET)"

# Comparação de Todos os Engines
ocr-compare:
	@echo "$(MAGENTA)📊 Comparando OCRs...$(RESET)"
	@if not exist "$(DATA_DIR)\ocr_test\ground_truth.json" ( \
		echo "$(RED)❌ Ground truth não encontrado!$(RESET)" && \
		echo "$(YELLOW)💡 Anote o ground truth em $(DATA_DIR)/ocr_test/ground_truth.json$(RESET)" && \
		exit 1 \
	)
	$(PYTHON) $(SCRIPTS_DIR)/ocr/benchmark_ocrs.py \
		--config $(CONFIG_DIR)/experiments/ocr_comparison.yaml \
		--output outputs/ocr_benchmarks/comparison \
		$(if $(PREP),--preprocessing $(PREP),) \
		$(if $(ENGINE),--engine $(ENGINE))
	@echo "$(GREEN)✅ Comparação concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/ocr_benchmarks/comparison/$(RESET)"
	@echo "$(CYAN)💡 Cada OCR usa seu pré-processamento otimizado: ppro-{engine}$(RESET)"
	@echo "$(CYAN)💡 Configs disponíveis: ppro-none, ppro-tesseract, ppro-easyocr, ppro-paddleocr, ppro-trocr, ppro-parseq$(RESET)"
	@echo "$(CYAN)💡 Use PREP=ppro-none para baseline sem pré-processamento$(RESET)"

# Benchmark Completo (todos os engines)
ocr-benchmark:
	@echo "$(MAGENTA)🏆 Benchmark completo de OCRs...$(RESET)"
	@echo "$(YELLOW)⚠️  Isso pode levar alguns minutos...$(RESET)"
	@echo ""
	@make ocr-test ENGINE=tesseract
	@make ocr-test ENGINE=easyocr
	@make ocr-test ENGINE=openocr
	@make ocr-test ENGINE=paddleocr
	@make ocr-test ENGINE=trocr
	@make ocr-test ENGINE=parseq
	@make ocr-compare
	@echo "$(GREEN)🎉 Benchmark completo!$(RESET)"

# =============================================================================
# 🔓 OpenOCR - Open-source High-Accuracy OCR
# =============================================================================
# Comandos para testar OpenOCR (backend ONNX ou PyTorch)
# ✅ Suporte para CPU e GPU (CUDA)

.PHONY: ocr-openocr ocr-openocr-quick ocr-openocr-benchmark

# Teste padrão do OpenOCR
ocr-openocr:
	@echo "$(BLUE)🔓 Testando OpenOCR (backend ONNX)...$(RESET)"
	@echo "$(CYAN)✨ Backend: ONNX (rápido e eficiente)$(RESET)"
	@make ocr-test ENGINE=openocr
	@echo "$(GREEN)✅ OpenOCR testado com sucesso!$(RESET)"
	@echo "$(CYAN)💡 Config: config/ocr/openocr.yaml$(RESET)"
	@echo "$(CYAN)📊 Relatório: outputs/ocr_benchmarks/openocr/report.html$(RESET)"

# Teste rápido do OpenOCR
ocr-openocr-quick:
	@echo "$(BLUE)⚡ Teste rápido OpenOCR...$(RESET)"
	@echo "$(CYAN)✨ Backend: ONNX (rápido e eficiente)$(RESET)"
	@echo "$(YELLOW)💡 Dica: Para teste completo use 'make ocr-openocr'$(RESET)"
	@make ocr-test ENGINE=openocr
	@echo "$(GREEN)✅ Teste rápido concluído!$(RESET)"
	@echo "$(CYAN)📊 Relatório: outputs/ocr_benchmarks/openocr/report.html$(RESET)"

# Benchmark completo do OpenOCR
ocr-openocr-benchmark:
	@echo "$(MAGENTA)🏆 Benchmark completo do OpenOCR...$(RESET)"
	@make ocr-openocr
	@echo ""
	@echo "$(GREEN)🎉 Benchmark OpenOCR concluído!$(RESET)"
	@echo "$(CYAN)📊 Métricas disponíveis:$(RESET)"
	@echo "   - Accuracy (taxa de acerto exata)"
	@echo "   - CER (Character Error Rate)"
	@echo "   - WER (Word Error Rate)"
	@echo "   - Tempo de processamento"
	@echo ""
	@echo "$(CYAN)💡 Compare com outros engines: make ocr-benchmark$(RESET)"

# =============================================================================
# 🤖 TrOCR - Transformer OCR com Normalização de Brilho
# =============================================================================
# Comandos para testar TrOCR (microsoft/trocr-base-printed)
# ✅ Integrado com normalização de brilho, CLAHE e remoção de sombras

.PHONY: ocr-trocr ocr-trocr-quick ocr-trocr-benchmark

# Teste padrão do TrOCR (com normalização de brilho)
ocr-trocr:
	@echo "$(BLUE)🤖 Testando TrOCR (microsoft/trocr-base-printed)...$(RESET)"
	@echo "$(CYAN)✨ Normalização de brilho: ATIVADA$(RESET)"
	@make ocr-test ENGINE=trocr
	@echo "$(GREEN)✅ TrOCR testado com sucesso!$(RESET)"
	@echo "$(CYAN)💡 Config: config/ocr/trocr.yaml$(RESET)"
	@echo "$(CYAN)📊 Relatório: outputs/ocr_benchmarks/trocr/report.html$(RESET)"

# Teste rápido do TrOCR (apenas primeiras 10 imagens)
ocr-trocr-quick:
	@echo "$(BLUE)⚡ Teste rápido TrOCR...$(RESET)"
	@echo "$(CYAN)✨ Normalização de brilho: ATIVADA$(RESET)"
	@echo "$(YELLOW)💡 Dica: Para teste completo use 'make ocr-trocr'$(RESET)"
	@make ocr-test ENGINE=trocr
	@echo "$(GREEN)✅ Teste rápido concluído!$(RESET)"
	@echo "$(CYAN)📊 Relatório: outputs/ocr_benchmarks/trocr/report.html$(RESET)"

# Benchmark completo do TrOCR (alias para ocr-trocr)
ocr-trocr-benchmark:
	@echo "$(MAGENTA)🏆 Benchmark completo do TrOCR...$(RESET)"
	@make ocr-trocr
	@echo ""
	@echo "$(GREEN)🎉 Benchmark TrOCR concluído!$(RESET)"
	@echo "$(CYAN)📊 Métricas disponíveis:$(RESET)"
	@echo "   - Accuracy (taxa de acerto exata)"
	@echo "   - CER (Character Error Rate)"
	@echo "   - WER (Word Error Rate)"
	@echo "   - Tempo de processamento"
	@echo ""
	@echo "$(CYAN)💡 Compare com outros engines: make ocr-benchmark$(RESET)"

# Validar normalização de brilho do TrOCR
.PHONY: ocr-trocr-validate-brightness

ocr-trocr-validate-brightness:
	@echo "$(BLUE)🔆 Validando normalização de brilho do TrOCR...$(RESET)"
	@echo "$(CYAN)Este teste valida:$(RESET)"
	@echo "   - Imagens muito brilhantes (brightness > 200)"
	@echo "   - Imagens muito escuras (brightness < 80)"
	@echo "   - Imagens com brilho adequado"
	@echo ""
	$(PYTHON) scripts/ocr/test_trocr_brightness.py
	@echo ""
	@echo "$(GREEN)✅ Validação concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/trocr_brightness_test/$(RESET)"

# Validação completa do PARSeq
.PHONY: ocr-parseq-validate

ocr-parseq-validate:
	@echo "$(BLUE)🔍 Validando implementação do PARSeq TINE...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/validate_parseq.py
	@echo "$(GREEN)✅ Validação concluída!$(RESET)"

# Comparar OCR com diferentes configurações de pré-processamento
.PHONY: ocr-compare-preprocessing

ocr-compare-preprocessing:
	@echo "$(MAGENTA)📊 Comparando OCRs com diferentes configurações de pré-processamento...$(RESET)"
	@echo ""
	@echo "$(BLUE)1/6 🔍 Testando SEM pré-processamento (baseline)...$(RESET)"
	@make ocr-compare PREP=ppro-none
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-none
	@echo ""
	@echo "$(BLUE)2/6 🔍 Testando pré-processamento TESSERACT...$(RESET)"
	@make ocr-compare PREP=ppro-tesseract
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-tesseract
	@echo ""
	@echo "$(BLUE)3/6 🔍 Testando pré-processamento EASYOCR...$(RESET)"
	@make ocr-compare PREP=ppro-easyocr
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-easyocr
	@echo ""
	@echo "$(BLUE)4/6 🔍 Testando pré-processamento PADDLEOCR...$(RESET)"
	@make ocr-compare PREP=ppro-paddleocr
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-paddleocr
	@echo ""
	@echo "$(BLUE)5/6 🔍 Testando pré-processamento TROCR...$(RESET)"
	@make ocr-compare PREP=ppro-trocr
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-trocr
	@echo ""
	@echo "$(BLUE)6/6 🔍 Testando pré-processamento PARSEQ...$(RESET)"
	@make ocr-compare PREP=ppro-parseq
	@move outputs\ocr_benchmarks\comparison outputs\ocr_benchmarks\comparison_ppro-parseq
	@echo ""
	@echo "$(GREEN)🎉 Comparação completa de pré-processamento concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados em: outputs/ocr_benchmarks/$(RESET)"
	@echo "   - comparison_ppro-none/       (sem pré-processamento - baseline)"
	@echo "   - comparison_ppro-tesseract/  (otimizado para Tesseract)"
	@echo "   - comparison_ppro-easyocr/    (otimizado para EasyOCR)"
	@echo "   - comparison_ppro-paddleocr/  (otimizado para PaddleOCR)"
	@echo "   - comparison_ppro-trocr/      (otimizado para TrOCR)"
	@echo "   - comparison_ppro-parseq/     (otimizado para PARSeq TINE)"

# Teste de Pré-processamento
prep-test:
ifndef CONFIG
	@echo "$(RED)❌ Especifique: make prep-test CONFIG=ppro-paddleocr$(RESET)"
	@echo "$(YELLOW)Configs disponíveis: ppro-none, ppro-tesseract, ppro-easyocr, ppro-paddleocr, ppro-trocr, ppro-parseq$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)🔍 Testando preprocessing $(CONFIG)...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/test_preprocessing.py \
		--config $(CONFIG_DIR)/preprocessing/$(CONFIG).yaml \
		--test-data $(DATA_DIR)/ocr_test \
		--visualize \
		--max-samples 10
	@echo "$(GREEN)✅ Teste de preprocessing concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/preprocessing_tests/$(CONFIG)/$(RESET)"

# Comparação de Configurações de Pré-processamento
prep-compare:
	@echo "$(MAGENTA)📊 Comparando configurações de preprocessing...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/test_preprocessing.py \
		--compare-all \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/preprocessing_tests \
		--visualize \
		--max-samples 10
	@echo "$(GREEN)✅ Comparação de preprocessing concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/preprocessing_tests/$(RESET)"
	@echo "$(CYAN)💡 Comparação inclui: ppro-none, ppro-tesseract, ppro-easyocr, ppro-paddleocr, ppro-trocr$(RESET)"

# Demonstração interativa de Pré-processamento (Novas funcionalidades)
prep-demo:
	@echo "$(MAGENTA)🎨 Demonstração de Pré-processamento...$(RESET)"
	@echo "$(CYAN)💡 Demonstra normalize_colors e sharpen em ação$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_preprocessing.py
	@echo "$(GREEN)✅ Demonstração concluída!$(RESET)"
	@echo "$(CYAN)📊 Resultados salvos em: outputs/preprocessing_demo/$(RESET)"

# =============================================================================
# 🚀 FULL PIPELINE (YOLO → OCR → Parse)
# =============================================================================
# Pipeline completo para detecção e extração de datas de validade

.PHONY: pipeline-test pipeline-run pipeline-batch pipeline-demo

# Testar pipeline com imagem de exemplo
pipeline-test:
	@echo "$(MAGENTA)🚀 Testando pipeline completo...$(RESET)"
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image data/sample.jpg \
		--config $(CONFIG_DIR)/pipeline/full_pipeline.yaml
	@echo "$(GREEN)✅ Pipeline testado!$(RESET)"

# Executar pipeline em uma imagem específica
pipeline-run:
ifndef IMAGE
	@echo "$(RED)❌ Especifique: make pipeline-run IMAGE=test.jpg$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)🔍 Processando $(IMAGE)...$(RESET)"
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image "$(IMAGE)" \
		--config $(CONFIG_DIR)/pipeline/full_pipeline.yaml \
		--save-crops
	@echo "$(GREEN)✅ Pipeline executada!$(RESET)"
	@echo "$(CYAN)📊 Resultados em: outputs/pipeline/visualizations/$(RESET)"

# Processar diretório completo
pipeline-batch:
ifndef DIR
	@echo "$(RED)❌ Especifique: make pipeline-batch DIR=data/test_images/$(RESET)"
	@exit 1
endif
	@echo "$(MAGENTA)� Processando diretório: $(DIR)$(RESET)"
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image-dir "$(DIR)" \
		--config $(CONFIG_DIR)/pipeline/full_pipeline.yaml
	@echo "$(GREEN)✅ Processamento em lote concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados em: outputs/pipeline/batch_summary.json$(RESET)"

# Demo interativo do pipeline
pipeline-demo:
	@echo "$(MAGENTA)🎬 Demonstração do Full Pipeline$(RESET)"
	@echo "$(CYAN)Processando imagens de teste...$(RESET)"
	$(PYTHON) scripts/pipeline/test_full_pipeline.py \
		--image-dir data/ocr_test/ \
		--config $(CONFIG_DIR)/pipeline/full_pipeline.yaml \
		--save-crops
	@echo "$(GREEN)✅ Demo concluída!$(RESET)"
	@echo "$(CYAN)📊 Verifique: outputs/pipeline/$(RESET)"

# Experimento: Comparação Completa de OCR
exp-ocr-comparison:
	@echo "$(MAGENTA)📊 Rodando experimento: Comparação OCR...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/experiments/run_ocr_comparison.py \
		--config $(CONFIG_DIR)/experiments/ocr_comparison.yaml
	@echo "$(GREEN)🎉 Experimento concluído!$(RESET)"

# Visualização de Resultados OCR
viz-ocr-results:
	@echo "$(BLUE)📈 Gerando visualizações OCR...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/visualize_results.py \
		--results outputs/ocr_benchmarks/comparison/comparison_summary.csv \
		--output outputs/visualizations/ocr_comparison.png \
		--type ocr
	@echo "$(GREEN)✅ Visualização gerada: outputs/visualizations/ocr_comparison.png$(RESET)"

# Visualização de Resultados de Pré-processamento
viz-preprocessing:
	@echo "$(BLUE)📈 Gerando visualizações preprocessing...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/visualize_results.py \
		--results outputs/preprocessing_tests/results.csv \
		--output outputs/visualizations/preprocessing_comparison.png \
		--type preprocessing
	@echo "$(GREEN)✅ Visualização gerada: outputs/visualizations/preprocessing_comparison.png$(RESET)"

# Workflow Completo OCR (para TCC)
workflow-ocr:
	@echo "$(MAGENTA)🎓 WORKFLOW COMPLETO OCR$(RESET)"
	@echo "$(CYAN)Este comando executará todo o fluxo de OCR automaticamente$(RESET)"
	@echo ""
	@echo "$(BLUE)1/5 🔧 Instalando engines...$(RESET)"
	@make ocr-setup
	@echo ""
	@echo "$(BLUE)2/5 📦 Preparando dataset...$(RESET)"
	@make ocr-prepare-data
	@echo ""
	@echo "$(BLUE)3/5 🧪 Executando benchmark...$(RESET)"
	@make ocr-compare
	@echo ""
	@echo "$(BLUE)4/5 🔍 Testando preprocessing...$(RESET)"
	@make prep-compare
	@echo ""
	@echo "$(BLUE)5/5 📊 Gerando visualizações...$(RESET)"
	@make viz-ocr-results
	@make viz-preprocessing
	@echo ""
	@echo "$(GREEN)🎉 WORKFLOW OCR CONCLUÍDO!$(RESET)"
	@echo "$(YELLOW)📊 Resultados em: outputs/ocr_benchmarks/ e outputs/preprocessing_tests/$(RESET)"
	@echo "$(YELLOW)📈 Visualizações em: outputs/visualizations/$(RESET)"

# Limpeza de Dados OCR
clean-ocr:
	@echo "$(RED)🧹 Limpando dados OCR...$(RESET)"
	@if exist "$(DATA_DIR)\ocr_test" rmdir /s /q "$(DATA_DIR)\ocr_test"
	@if exist "outputs\ocr_benchmarks" rmdir /s /q "outputs\ocr_benchmarks"
	@if exist "outputs\preprocessing_tests" rmdir /s /q "outputs\preprocessing_tests"
	@echo "$(GREEN)✅ Dados OCR limpos!$(RESET)"




# Workflow completo para TCC - SEGMENTAÇÃO ⭐
workflow-tcc:
	@echo "$(MAGENTA)🎓 WORKFLOW COMPLETO TCC - SEGMENTAÇÃO POLIGONAL$(RESET)"
	@echo "$(CYAN)Este comando executará todo o fluxo do TCC automaticamente$(RESET)"
	@echo ""
ifndef INPUT
	@echo "$(RED)❌ Erro: Especifique INPUT=caminho_dos_dados_raw$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)1/6 📊 Processando dados SEGMENTAÇÃO...$(RESET)"
	make process-auto INPUT=$(INPUT)
	@echo "$(BLUE)2/6 🧪 Teste rápido SEGMENTAÇÃO...$(RESET)"
	make train-quick
	@echo "$(BLUE)3/6 🚀 Treinando modelos SEGMENTAÇÃO finais...$(RESET)"
	make train-final-nano
	make train-final-small
	make train-final-medium
	@echo "$(BLUE)4/6 📦 Treinando DETECÇÃO (comparação)...$(RESET)"
	make train-final-detect-small
	@echo "$(BLUE)5/6 📈 Gerando comparação...$(RESET)"
	make compare-final
	@echo "$(BLUE)6/6 📝 Gerando relatório...$(RESET)"
	make generate-report
	@echo "$(GREEN)🎉 WORKFLOW TCC CONCLUÍDO!$(RESET)"
	@echo "$(YELLOW)📊 Resultados em: experiments/$(RESET)"
	@echo "$(YELLOW)📈 Comparação: experiments/final_comparison.png$(RESET)"

# Workflow alternativo - DETECÇÃO
workflow-tcc-detect:
	@echo "$(MAGENTA)🎓 WORKFLOW TCC - DETECÇÃO$(RESET)"
	@echo ""
ifndef INPUT
	@echo "$(RED)❌ Erro: Especifique INPUT=caminho_dos_dados_raw$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)1/5 📊 Processando dados DETECÇÃO...$(RESET)"
	make process-detect INPUT=$(INPUT)
	@echo "$(BLUE)2/5 🧪 Teste rápido DETECÇÃO...$(RESET)"
	make train-quick-detect
	@echo "$(BLUE)3/5 🚀 Treinando modelos DETECÇÃO finais...$(RESET)"
	make train-final-detect-nano
	make train-final-detect-small
	make train-final-detect-medium
	@echo "$(BLUE)4/5 📈 Gerando comparação...$(RESET)"
	make compare-final
	@echo "$(BLUE)5/5 📝 Gerando relatório...$(RESET)"
	make generate-report
	@echo "$(GREEN)🎉 WORKFLOW TCC DETECÇÃO CONCLUÍDO!$(RESET)"

# =============================================================================
# 🚀 ENHANCED PARSeq - Pipeline Robusto com Multi-linha, Ensemble e Reranking
# =============================================================================
# Pipeline completo com:
# - Line detection (detecção e separação de linhas)
# - Geometric normalization (deskew, perspective)
# - Photometric normalization (CLAHE, denoise, shadow removal)
# - Ensemble de variantes (múltiplas alturas, CLAHE, denoise)
# - Reranking inteligente (confidence, dict match, consensus)
# - Contextual postprocessing (ambiguity mapping, format fixing)

.PHONY: ocr-enhanced ocr-enhanced-demo ocr-enhanced-batch ocr-enhanced-ablation
.PHONY: ocr-enhanced-fast ocr-enhanced-quality ocr-enhanced-experiment
.PHONY: ocr-enhanced-finetune ocr-enhanced-finetune-prepare ocr-enhanced-eval

# ========================================
# 1. DEMO & QUICK TESTS
# ========================================

# Demo interativo do Enhanced PARSeq
ocr-enhanced-demo:
	@echo "$(MAGENTA)🚀 Enhanced PARSeq - Demo Interativo$(RESET)"
	@echo "$(CYAN)Pipeline: Line Detection → Normalization → Ensemble → Reranking → Postprocessing$(RESET)"
ifndef IMAGE
	@echo "$(YELLOW)⚠️ Usando imagem de teste padrão...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--mode demo
else
	@echo "$(GREEN)📷 Processando: $(IMAGE)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--image "$(IMAGE)" \
		--mode demo \
		--visualize
endif
	@echo "$(GREEN)✅ Demo concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados salvos em: outputs/enhanced_parseq/demo/$(RESET)"

# Teste simples com configuração balanceada (padrão)
ocr-enhanced:
	@echo "$(BLUE)🔤 Testando Enhanced PARSeq (modo balanceado)...$(RESET)"
	$(PYTHON) -m src.ocr.evaluator \
		--engine parseq_enhanced \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/ocr_benchmarks/parseq_enhanced
	@echo "$(GREEN)✅ Enhanced PARSeq testado com sucesso!$(RESET)"
	@echo ""
	@echo "$(CYAN)📊 RESULTADOS DETALHADOS GERADOS:$(RESET)"
	@echo "$(CYAN)  📄 HTML Report: outputs/ocr_benchmarks/parseq_enhanced/report.html$(RESET)"
	@echo "$(CYAN)  📝 Markdown Report: outputs/ocr_benchmarks/parseq_enhanced/report.md$(RESET)"
	@echo "$(CYAN)  📈 Estatísticas JSON: outputs/ocr_benchmarks/parseq_enhanced/statistics.json$(RESET)"
	@echo "$(CYAN)  📊 Visualizações: outputs/ocr_benchmarks/parseq_enhanced/*.png$(RESET)"
	@echo ""
	@echo "$(MAGENTA)💡 Análises avançadas incluídas:$(RESET)"
	@echo "   ✅ Estatísticas básicas (exact match, CER, confidence)"
	@echo "   📊 Análise de erros por categoria (perfect, low, medium, high)"
	@echo "   🔤 Matriz de confusão de caracteres"
	@echo "   📝 Análise em nível de palavras"
	@echo "   📏 Impacto do comprimento do texto"
	@echo "   ⏱️  Análise de desempenho e tempo"
	@echo "   📈 Correlação confiança vs precisão"
	@echo "   🎯 Dashboard de performance completo"

# Teste rápido (sem ensemble, para velocidade)
ocr-enhanced-fast:
	@echo "$(BLUE)⚡ Enhanced PARSeq - Modo Rápido$(RESET)"
	@echo "$(YELLOW)⚠️ Ensemble desabilitado para maior velocidade$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--preset fast \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/ocr_benchmarks/parseq_enhanced_fast
	@echo "$(GREEN)✅ Teste rápido concluído!$(RESET)"

# Teste com máxima qualidade (ensemble completo, modelo large)
ocr-enhanced-quality:
	@echo "$(BLUE)🏆 Enhanced PARSeq - Modo Alta Qualidade$(RESET)"
	@echo "$(CYAN)Configuração: modelo LARGE + 5 variantes + reranking$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--preset high_quality \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/ocr_benchmarks/parseq_enhanced_quality
	@echo "$(GREEN)✅ Teste de qualidade concluído!$(RESET)"
	@echo "$(YELLOW)⚠️ Modo lento, mas máxima precisão$(RESET)"

# 🔍 Diagnóstico completo (salva todas as etapas intermediárias)
ocr-diagnose:
ifndef IMAGE
	@echo "$(RED)❌ Especifique: make ocr-diagnose IMAGE=caminho/imagem.jpg$(RESET)"
	@echo "$(YELLOW)Exemplo: make ocr-diagnose IMAGE=data/ocr_test/images/IMG001.jpg GT=\"25/10/2025\"$(RESET)"
	@exit 1
endif
	@echo "$(MAGENTA)🔍 DIAGNÓSTICO COMPLETO - Enhanced PARSeq$(RESET)"
	@echo "$(CYAN)Imagem: $(IMAGE)$(RESET)"
	@echo "$(CYAN)Ground Truth: $(GT)$(RESET)"
	@echo "$(CYAN)Salvando todas as etapas intermediárias...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/diagnose_enhanced_parseq.py \
		--image "$(IMAGE)" \
		$(if $(GT),--ground-truth "$(GT)",) \
		--output outputs/ocr_debug
	@echo "$(GREEN)✅ Diagnóstico concluído!$(RESET)"
	@echo "$(CYAN)📁 Imagens intermediárias salvas em: outputs/ocr_debug/$(RESET)"
	@echo "$(YELLOW)💡 Analise todas as etapas para identificar o problema$(RESET)"

# ========================================
# 2. BATCH PROCESSING
# ========================================

# Processar diretório completo
ocr-enhanced-batch:
ifndef DIR
	@echo "$(RED)❌ Especifique: make ocr-enhanced-batch DIR=caminho/do/diretorio$(RESET)"
	@echo "$(YELLOW)Exemplo: make ocr-enhanced-batch DIR=data/ocr_test$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)📦 Enhanced PARSeq - Processamento em Lote$(RESET)"
	@echo "$(CYAN)Diretório: $(DIR)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--mode batch \
		--input-dir "$(DIR)" \
		--output outputs/enhanced_parseq/batch \
		--visualize \
		
		--save-metrics
	@echo "$(GREEN)✅ Processamento em lote concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/enhanced_parseq/batch/$(RESET)"

# ========================================
# 3. ABLATION STUDY
# ========================================

# Estudo de ablação completo
ocr-enhanced-ablation:
	@echo "$(MAGENTA)🔬 Enhanced PARSeq - Estudo de Ablação$(RESET)"
	@echo "$(CYAN)Testando todas as combinações de componentes...$(RESET)"
	@echo ""
	@echo "$(YELLOW)Componentes:$(RESET)"
	@echo "  1️⃣ Line Detection"
	@echo "  2️⃣ Geometric Normalization"
	@echo "  3️⃣ Photometric Normalization"
	@echo "  4️⃣ Ensemble + Reranking"
	@echo "  5️⃣ Contextual Postprocessing"
	@echo ""
	$(PYTHON) $(SCRIPTS_DIR)/ocr/demo_enhanced_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--mode ablation \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/enhanced_parseq/ablation
	@echo ""
	@echo "$(GREEN)✅ Ablation study concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados detalhados: outputs/enhanced_parseq/ablation/$(RESET)"
	@echo "$(CYAN)📈 Gráfico de ablação: outputs/enhanced_parseq/ablation/ablation_results.png$(RESET)"

# Ablação rápida (subset de combinações)
ocr-enhanced-ablation-quick:
	@echo "$(BLUE)🔬 Enhanced PARSeq - Ablação Rápida$(RESET)"
	@echo "$(YELLOW)Testando combinações principais...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/run_ablation.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/enhanced_parseq/ablation_quick \
		--quick-mode
	
	@echo "$(GREEN)✅ Ablação rápida concluída!$(RESET)"

# ========================================
# 4. EXPERIMENTOS COMPARATIVOS
# ========================================

# Comparar Enhanced vs Baseline
ocr-enhanced-vs-baseline:
	@echo "$(MAGENTA)📊 Comparando Enhanced PARSeq vs Baseline$(RESET)"
	@echo ""
	@echo "$(BLUE)1/2 Rodando Baseline (PARSeq simples)...$(RESET)"
	@make ocr-parseq-base
	@echo ""
	@echo "$(BLUE)2/2 Rodando Enhanced PARSeq...$(RESET)"
	@make ocr-enhanced
	@echo ""
	@echo "$(CYAN)📊 Gerando comparação...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/compare_enhanced_vs_baseline.py \
		--baseline outputs/ocr_benchmarks/parseq_base \
		--enhanced outputs/ocr_benchmarks/parseq_enhanced \
		--output outputs/enhanced_parseq/comparison
	@echo "$(GREEN)✅ Comparação concluída!$(RESET)"
	@echo "$(CYAN)📈 Gráficos: outputs/enhanced_parseq/comparison/$(RESET)"

# Experimento completo (todos os presets)
ocr-enhanced-experiment:
	@echo "$(MAGENTA)🧪 Enhanced PARSeq - Experimento Completo$(RESET)"
	@echo "$(CYAN)Testando todos os presets: fast, balanced, high_quality$(RESET)"
	@echo ""
	$(PYTHON) $(SCRIPTS_DIR)/ocr/run_experiment.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/enhanced_parseq/experiments \
		--presets fast balanced high_quality \
		--compare-all
	@echo ""
	@echo "$(GREEN)✅ Experimento completo concluído!$(RESET)"
	@echo "$(CYAN)📊 Resultados: outputs/enhanced_parseq/experiments/$(RESET)"

# ========================================
# 5. FINE-TUNING
# ========================================

# Preparar dados para fine-tuning
ocr-enhanced-finetune-prepare:
	@echo "$(BLUE)📦 Preparando dados para fine-tuning...$(RESET)"
ifndef TRAIN_DIR
	@echo "$(RED)❌ Especifique: make ocr-enhanced-finetune-prepare TRAIN_DIR=caminho VAL_DIR=caminho$(RESET)"
	@echo "$(YELLOW)Exemplo: make ocr-enhanced-finetune-prepare TRAIN_DIR=data/ocr_train VAL_DIR=data/ocr_val$(RESET)"
	@exit 1
endif
ifndef VAL_DIR
	@echo "$(RED)❌ Especifique VAL_DIR=caminho$(RESET)"
	@exit 1
endif
	@echo "$(CYAN)Train: $(TRAIN_DIR)$(RESET)"
	@echo "$(CYAN)Val: $(VAL_DIR)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/prepare_finetune_data.py \
		--train-dir "$(TRAIN_DIR)" \
		--val-dir "$(VAL_DIR)" \
		--output data/ocr_finetuning \
		--format json \
		--validate
	@echo "$(GREEN)✅ Dados preparados!$(RESET)"
	@echo "$(CYAN)📂 Saída: data/ocr_finetuning/$(RESET)"

# Fine-tuning do modelo
ocr-enhanced-finetune:
	@echo "$(MAGENTA)🎓 Fine-tuning Enhanced PARSeq$(RESET)"
	@echo "$(CYAN)Configuração: config/ocr/parseq_enhanced_full.yaml$(RESET)"
ifndef TRAIN_DATA
	@echo "$(YELLOW)⚠️ Usando dataset padrão: data/ocr_finetuning$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/finetune_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--train-data data/ocr_finetuning/train \
		--val-data data/ocr_finetuning/val \
		--output models/parseq_finetuned \
		--epochs 50 \
		--batch-size 32 \
		--learning-rate 1e-4
else
	@echo "$(CYAN)Train data: $(TRAIN_DATA)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/finetune_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--train-data "$(TRAIN_DATA)" \
		--val-data "$(VAL_DATA)" \
		--output models/parseq_finetuned \
		--epochs 50 \
		--batch-size 32 \
		--learning-rate 1e-4
endif
	@echo "$(GREEN)✅ Fine-tuning concluído!$(RESET)"
	@echo "$(CYAN)📂 Modelo: models/parseq_finetuned/$(RESET)"

# Fine-tuning rápido (teste)
ocr-enhanced-finetune-test:
	@echo "$(BLUE)🧪 Fine-tuning de Teste (10 épocas)$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/finetune_parseq.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--train-data data/ocr_finetuning/train \
		--val-data data/ocr_finetuning/val \
		--output models/parseq_finetuned_test \
		--epochs 10 \
		--batch-size 16 \
		--learning-rate 1e-4
	@echo "$(GREEN)✅ Teste de fine-tuning concluído!$(RESET)"

# ========================================
# 6. EVALUATION & METRICS
# ========================================

# Avaliar modelo fine-tuned
ocr-enhanced-eval:
ifndef MODEL
	@echo "$(RED)❌ Especifique: make ocr-enhanced-eval MODEL=caminho/do/modelo$(RESET)"
	@echo "$(YELLOW)Exemplo: make ocr-enhanced-eval MODEL=models/parseq_finetuned/best.pt$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)📊 Avaliando modelo fine-tuned...$(RESET)"
	@echo "$(CYAN)Modelo: $(MODEL)$(RESET)"
	$(PYTHON) -m src.ocr.evaluator \
		--engine parseq_enhanced \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--model-path "$(MODEL)" \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/ocr_benchmarks/parseq_finetuned \
		--detailed-metrics
	@echo "$(GREEN)✅ Avaliação concluída!$(RESET)"
	@echo "$(CYAN)📊 Métricas: outputs/ocr_benchmarks/parseq_finetuned/$(RESET)"

# Comparar modelo original vs fine-tuned
ocr-enhanced-compare-finetuned:
ifndef MODEL
	@echo "$(RED)❌ Especifique: make ocr-enhanced-compare-finetuned MODEL=caminho/do/modelo$(RESET)"
	@exit 1
endif
	@echo "$(MAGENTA)📊 Comparando Original vs Fine-tuned$(RESET)"
	@echo ""
	@echo "$(BLUE)1/2 Avaliando modelo original...$(RESET)"
	@make ocr-enhanced
	@echo ""
	@echo "$(BLUE)2/2 Avaliando modelo fine-tuned...$(RESET)"
	@make ocr-enhanced-eval MODEL="$(MODEL)"
	@echo ""
	@echo "$(CYAN)📊 Gerando comparação...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/compare_models.py \
		--baseline outputs/ocr_benchmarks/parseq_enhanced \
		--finetuned outputs/ocr_benchmarks/parseq_finetuned \
		--output outputs/enhanced_parseq/finetuned_comparison
	@echo "$(GREEN)✅ Comparação concluída!$(RESET)"

# ========================================
# 7. SYNTHETIC DATA GENERATION
# ========================================

# Gerar dados sintéticos para fine-tuning
ocr-enhanced-generate-synthetic:
	@echo "$(BLUE)🎨 Gerando dados sintéticos...$(RESET)"
ifndef NUM
	@echo "$(YELLOW)⚠️ Usando quantidade padrão: 10000 amostras$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/generate_synthetic_data.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--output data/ocr_synthetic \
		--num-samples 10000
else
	@echo "$(CYAN)Gerando $(NUM) amostras...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/generate_synthetic_data.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced_full.yaml \
		--output data/ocr_synthetic \
		--num-samples $(NUM)
endif
	@echo "$(GREEN)✅ Dados sintéticos gerados!$(RESET)"
	@echo "$(CYAN)📂 Saída: data/ocr_synthetic/$(RESET)"

# ========================================
# 8. ANÁLISE & VISUALIZAÇÕES
# ========================================

# Visualizar componentes do pipeline
ocr-enhanced-visualize:
ifndef IMAGE
	@echo "$(RED)❌ Especifique: make ocr-enhanced-visualize IMAGE=caminho/da/imagem$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)🎨 Visualizando pipeline step-by-step...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/visualize_pipeline.py \
		--config $(CONFIG_DIR)/ocr/parseq_enhanced.yaml \
		--image "$(IMAGE)" \
		--output outputs/enhanced_parseq/visualizations \
		--show-all-steps
	@echo "$(GREEN)✅ Visualizações geradas!$(RESET)"
	@echo "$(CYAN)📂 Saída: outputs/enhanced_parseq/visualizations/$(RESET)"

# Análise de erros detalhada
ocr-enhanced-error-analysis:
	@echo "$(BLUE)🔍 Análise de erros detalhada...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/analyze_errors.py \
		--results outputs/ocr_benchmarks/parseq_enhanced \
		--test-data $(DATA_DIR)/ocr_test \
		--output outputs/enhanced_parseq/error_analysis \
		--categorize-errors \
		--visualize
	@echo "$(GREEN)✅ Análise de erros concluída!$(RESET)"
	@echo "$(CYAN)📊 Relatório: outputs/enhanced_parseq/error_analysis/$(RESET)"

# ========================================
# 9. WORKFLOW COMPLETO
# ========================================

# Workflow completo: setup → test → experiment → report
workflow-enhanced-parseq:
	@echo "$(MAGENTA)🎓 WORKFLOW COMPLETO - Enhanced PARSeq$(RESET)"
	@echo "$(CYAN)═══════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(BLUE)1/6 🔧 Setup: Baixando modelos...$(RESET)"
	@make ocr-parseq-setup
	@echo ""
	@echo "$(BLUE)2/6 📦 Preparando dataset OCR...$(RESET)"
	@make ocr-prepare-data
	@echo ""
	@echo "$(BLUE)3/6 🧪 Teste rápido (demo)...$(RESET)"
	@make ocr-enhanced-demo
	@echo ""
	@echo "$(BLUE)4/6 📊 Estudo de ablação...$(RESET)"
	@make ocr-enhanced-ablation
	@echo ""
	@echo "$(BLUE)5/6 🔬 Comparação vs baseline...$(RESET)"
	@make ocr-enhanced-vs-baseline
	@echo ""
	@echo "$(BLUE)6/6 📈 Gerando relatório final...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/generate_report.py \
		--ablation outputs/enhanced_parseq/ablation \
		--comparison outputs/enhanced_parseq/comparison \
		--output outputs/enhanced_parseq/final_report.pdf
	@echo ""
	@echo "$(GREEN)🎉 WORKFLOW COMPLETO CONCLUÍDO!$(RESET)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(CYAN)📊 Resultados:$(RESET)"
	@echo "   • Demo: outputs/enhanced_parseq/demo/"
	@echo "   • Ablation: outputs/enhanced_parseq/ablation/"
	@echo "   • Comparison: outputs/enhanced_parseq/comparison/"
	@echo "   • 📄 Relatório Final: outputs/enhanced_parseq/final_report.pdf"

# Workflow de fine-tuning completo
workflow-enhanced-finetune:
	@echo "$(MAGENTA)🎓 WORKFLOW FINE-TUNING - Enhanced PARSeq$(RESET)"
	@echo ""
ifndef TRAIN_DIR
	@echo "$(RED)❌ Especifique: make workflow-enhanced-finetune TRAIN_DIR=... VAL_DIR=...$(RESET)"
	@exit 1
endif
	@echo "$(BLUE)1/5 📦 Preparando dados...$(RESET)"
	@make ocr-enhanced-finetune-prepare TRAIN_DIR="$(TRAIN_DIR)" VAL_DIR="$(VAL_DIR)"
	@echo ""
	@echo "$(BLUE)2/5 🎨 Gerando dados sintéticos (opcional)...$(RESET)"
	@make ocr-enhanced-generate-synthetic NUM=5000
	@echo ""
	@echo "$(BLUE)3/5 🎓 Fine-tuning do modelo...$(RESET)"
	@make ocr-enhanced-finetune
	@echo ""
	@echo "$(BLUE)4/5 📊 Avaliando modelo fine-tuned...$(RESET)"
	@make ocr-enhanced-compare-finetuned MODEL=models/parseq_finetuned/best.pt
	@echo ""
	@echo "$(BLUE)5/5 📈 Gerando relatório...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/ocr/generate_finetune_report.py \
		--comparison outputs/enhanced_parseq/finetuned_comparison \
		--output outputs/enhanced_parseq/finetune_report.pdf
	@echo ""
	@echo "$(GREEN)🎉 WORKFLOW FINE-TUNING CONCLUÍDO!$(RESET)"
	@echo "$(CYAN)📄 Relatório: outputs/enhanced_parseq/finetune_report.pdf$(RESET)"

# ========================================
# 10. HELP & DOCUMENTATION
# ========================================

# Help específico para Enhanced PARSeq
help-enhanced-parseq:
	@echo "$(CYAN)════════════════════════════════════════════════$(RESET)"
	@echo "$(MAGENTA)  🚀 Enhanced PARSeq - Guia de Comandos$(RESET)"
	@echo "$(CYAN)════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(GREEN)📌 TESTES RÁPIDOS:$(RESET)"
	@echo "  ocr-enhanced-demo              Demo interativo com imagem de teste"
	@echo "  ocr-enhanced                   Teste padrão (modo balanceado)"
	@echo "  ocr-enhanced-fast              Modo rápido (sem ensemble)"
	@echo "  ocr-enhanced-quality           Modo alta qualidade (lento)"
	@echo ""
	@echo "$(GREEN)📦 BATCH PROCESSING:$(RESET)"
	@echo "  ocr-enhanced-batch DIR=...     Processar diretório completo"
	@echo ""
	@echo "$(GREEN)🔬 EXPERIMENTOS:$(RESET)"
	@echo "  ocr-enhanced-ablation          Estudo de ablação completo"
	@echo "  ocr-enhanced-vs-baseline       Comparar vs baseline"
	@echo "  ocr-enhanced-experiment        Experimento com todos presets"
	@echo ""
	@echo "$(GREEN)🎓 FINE-TUNING:$(RESET)"
	@echo "  ocr-enhanced-finetune-prepare  Preparar dados para fine-tuning"
	@echo "  ocr-enhanced-finetune          Fine-tuning do modelo"
	@echo "  ocr-enhanced-eval MODEL=...    Avaliar modelo fine-tuned"
	@echo "  ocr-enhanced-generate-synthetic Gerar dados sintéticos"
	@echo ""
	@echo "$(GREEN)📊 ANÁLISE:$(RESET)"
	@echo "  ocr-enhanced-visualize IMAGE=  Visualizar pipeline step-by-step"
	@echo "  ocr-enhanced-error-analysis    Análise detalhada de erros"
	@echo ""
	@echo "$(GREEN)🎯 WORKFLOWS:$(RESET)"
	@echo "  workflow-enhanced-parseq       Workflow completo (demo→ablation→comparação)"
	@echo "  workflow-enhanced-finetune     Workflow fine-tuning completo"
	@echo ""
	@echo "$(YELLOW)💡 EXEMPLOS:$(RESET)"
	@echo "  make ocr-enhanced-demo IMAGE=test.jpg"
	@echo "  make ocr-enhanced-batch DIR=data/ocr_test"
	@echo "  make ocr-enhanced-finetune TRAIN_DATA=data/train VAL_DATA=data/val"
	@echo ""
	@echo "$(CYAN)📚 Documentação:$(RESET)"
	@echo "  • docs/PARSEQ_ENHANCED_GUIDE.md"
	@echo "  • docs/IMPLEMENTATION_CHECKLIST.md"
	@echo "  • docs/CODE_EXAMPLES.md"
	@echo "  • docs/FAQ_ENHANCED_PARSEQ.md"
	@echo "$(CYAN)════════════════════════════════════════════════$(RESET)"