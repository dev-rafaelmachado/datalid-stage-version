# 🧹 Guia de Limpeza do Projeto Datalid 3.0

## 📋 Resumo Executivo

Este documento lista todas as ações necessárias para limpar o projeto antes de enviar para o professor. O projeto foi desenvolvido de forma iterativa ("vibe coding"), resultando em duplicações, arquivos vazios e estrutura que pode ser simplificada.

---

## 🗑️ PRIORIDADE ALTA - Arquivos para DELETAR

### Arquivos Vazios (Deletar Imediatamente)
```
scripts/evaluation/compare_learning_curves.py          # VAZIO
scripts/inference/predict_latest.py                    # VAZIO
scripts/inference/predict_yolo.py                      # VAZIO
scripts/inference/test_inference.py                    # VAZIO
scripts/ocr/test_tesseract.py                         # VAZIO
scripts/ocr/test_trocr_brightness.py                  # VAZIO
scripts/pipeline/evaluate_pipeline.py                  # VAZIO
scripts/pipeline/test_full_pipeline.py                 # VAZIO
```

### Arquivos Duplicados (Manter apenas 1)

#### Scripts de Teste OpenOCR - **DELETAR**:
```
scripts/test_openocr.py                               # Duplicado
scripts/ocr/test_openocr.py                          # MANTER este
```

#### Scripts de Predição - **CONSOLIDAR**:
```
scripts/inference/predict_latest_model.py             # Funcionalidade similar
scripts/inference/predict_single.py                   # MANTER este (mais completo)
```

### Scripts de Teste/Debug Temporários - **DELETAR**:
```
scripts/infra/yaml_test.py                            # Teste pontual de YAML
scripts/ocr/debug_parseq.py                          # Debug temporário
scripts/ocr/quick_test_parseq.py                     # Teste rápido
scripts/ocr/quick_test_enhanced.py                   # Teste rápido
scripts/ocr/validate_parseq.py                       # Validação pontual
scripts/ocr/test_parseq_fixed.py                     # Teste após fix
scripts/ocr/test_parseq_real_images.py              # Teste específico
```

### Scripts de "Exemplo" - **CONSOLIDAR EM 1**:
Manter apenas `exemplos_enhanced.py` como exemplo único de uso
```
scripts/ocr/exemplo_parseq.py                         # DELETAR
scripts/ocr/exemplo_openocr.py                        # DELETAR
scripts/ocr/exemplo_analise_detalhada.py             # DELETAR
scripts/ocr/exemplos_enhanced.py                      # MANTER (renomear para exemplo_uso_completo.py)
```

---

## 🔄 PRIORIDADE MÉDIA - Arquivos para CONSOLIDAR

### Scripts de Diagnóstico OCR (3 arquivos similares):
**Manter apenas:** `diagnose_enhanced_parseq.py` (mais completo)
**Deletar:**
```
scripts/ocr/diagnose_ocr_problems.py                  # Funcionalidade similar
```

### Scripts de Benchmark (podem ser 1 só):
**Consolidar em:** `benchmark_ocrs.py` (principal)
**Revisar necessidade de:**
```
scripts/ocr/benchmark_parseq_enhanced.py              # Específico para PARSeq
scripts/ocr/compare_parseq_models.py                  # Comparação de variantes
```

### Scripts de Demonstração:
**Consolidar em 1 exemplo principal:**
```
scripts/ocr/demo_enhanced_parseq.py                   # MANTER
scripts/ocr/demo_preprocessing.py                     # Pode ser incorporado
```

---

## 📁 ESTRUTURA DE PASTAS - Simplificações

### `scripts/api/` - **3 arquivos para rodar API**:
Problema: Confuso ter 3 formas de rodar
```
scripts/api/run_api.py                                # MANTER (principal)
scripts/api/start_server.py                          # DELETAR (duplicado)
```
**Solução:** Manter apenas `run_api.py`

### `scripts/data/` - Revisar necessidade:
Scripts que talvez não sejam mais usados:
```
scripts/data/process_with_fraction.py                 # Verificar se ainda usado
scripts/data/annotate_ground_truth.py                 # Ferramenta de anotação manual
```

---

## 🧪 SCRIPTS DE TESTE - Organizar

### Testes que devem virar testes unitários (pytest):
Muitos scripts em `scripts/` deveriam estar em `tests/`:
```
scripts/setup/test_cuda.py                            # → tests/test_cuda.py
scripts/utils/test_ocr_statistics.py                  # → tests/test_ocr_statistics.py
scripts/ocr/test_ocr_module.py                        # → tests/test_ocr_module.py
scripts/ocr/test_preprocessing.py                     # → tests/test_preprocessing.py
```

---

## 📄 ARQUIVOS DE CONFIGURAÇÃO - Revisar

### Configs YAML duplicados ou não usados:
Verifique em `config/`:
- Há muitos presets de preprocessing (`ppro-*.yaml`) - documentar quais são usados
- Múltiplas configs de OCR engines - consolidar se possível

### Configs de experimentos:
```
config/experiments/ocr_comparison.yaml                # Verificar se usado
config/pipeline/pipeline_evaluation.yaml              # Verificar se usado
```

---

## 🏗️ CÓDIGO FONTE - Melhorias

### `src/core/` - 3 gerenciadores de config:
Há sobreposição entre:
```
src/core/config.py           # Config com Pydantic
src/core/config_manager.py   # Gerenciador YAML
src/core/config_loader.py    # Outro loader YAML
```
**Ação:** Documentar claramente a diferença ou consolidar

### `src/ocr/` - Muitas engines:
Se algumas engines não performaram bem, considere remover:
- Tesseract (se não usado na versão final)
- TrOCR (se não usado na versão final)
- EasyOCR (se não usado na versão final)

**Manter apenas as engines que foram usadas nos resultados finais**

---

## 📊 OUTPUTS E DADOS - Limpar

### Diretórios para revisar:
```
outputs/                     # Limpar resultados antigos/intermediários
logs/                        # Limpar logs antigos
experiments/                 # Manter apenas experimentos relevantes para o TCC
e2e_results/                # Verificar se necessário
models/easyocr/             # Modelos baixados (não versionar se grandes)
```

### Arquivos de modelo na raiz:
```
yolov8m-seg.pt              # Modelo pré-treinado - pode não precisar versionar
yolov8n-seg.pt              # Modelo pré-treinado
yolov8n.pt                  # Modelo pré-treinado
yolov8s-seg.pt              # Modelo pré-treinado
```
**Ação:** Adicionar ao `.gitignore` e documentar onde baixar

---

## 📚 DOCUMENTAÇÃO - Consolidar

### Docs duplicados/desatualizados:
A pasta `docs/` tem muitos arquivos:
```
docs/update/                 # Parece ser histórico de atualizações
```
**Ação:** 
1. Manter apenas docs atualizados e relevantes
2. Arquivar histórico se necessário
3. Criar um `USAGE.md` simples para o professor

---

## ✅ CHECKLIST DE AÇÕES RECOMENDADAS

### Fase 1 - Limpeza Rápida (30 min)
- [ ] Deletar todos os arquivos vazios (8 arquivos)
- [ ] Deletar scripts duplicados óbvios (3-4 arquivos)
- [ ] Deletar scripts de teste temporários (7 arquivos)
- [ ] Adicionar `.pt` files ao `.gitignore`
- [ ] Limpar pasta `outputs/` de resultados antigos

### Fase 2 - Consolidação (1-2h)
- [ ] Consolidar scripts de exemplo em 1 arquivo bem documentado
- [ ] Consolidar scripts de diagnóstico (manter 1)
- [ ] Mover scripts de teste para pasta `tests/`
- [ ] Revisar e documentar configs YAML (quais são usados)
- [ ] Consolidar ou documentar os 3 config managers

### Fase 3 - Documentação (1h)
- [ ] Criar `USAGE.md` simples com exemplos principais
- [ ] Atualizar `README.md` com estrutura final
- [ ] Documentar qual OCR engine foi usado na versão final
- [ ] Criar `SETUP.md` com instruções de instalação clara
- [ ] Adicionar comentários nos scripts principais

### Fase 4 - Revisão Final (30 min)
- [ ] Rodar os scripts principais para garantir que funcionam
- [ ] Verificar se `requirements.txt` está completo
- [ ] Testar API com `python scripts/api/run_api.py`
- [ ] Testar pipeline completo
- [ ] Verificar se não há imports quebrados

---

## 🎯 ESTRUTURA FINAL RECOMENDADA

```
datalid3.0/
├── README.md                    # Overview do projeto
├── SETUP.md                     # Como instalar e rodar
├── USAGE.md                     # Exemplos de uso principais
├── requirements.txt             # Dependências
├── docker-compose.yml          
├── Dockerfile                  
├── Makefile                    
│
├── config/                      # Configs YAML organizados
│   ├── config.yaml              # Config principal
│   ├── ocr/                     # Configs de OCR (só os usados)
│   └── yolo/                    # Configs YOLO
│
├── src/                         # Código fonte principal
│   ├── api/                     # API REST
│   ├── core/                    # Núcleo do sistema
│   ├── data/                    # Processamento de dados
│   ├── ocr/                     # Módulo OCR
│   ├── pipeline/                # Pipelines completos
│   ├── utils/                   # Utilitários
│   └── yolo/                    # Módulo YOLO
│
├── scripts/                     # Scripts executáveis
│   ├── api/
│   │   └── run_api.py          # ÚNICO script para rodar API
│   ├── data/
│   │   ├── convert_dataset.py
│   │   └── prepare_ocr_dataset.py
│   ├── evaluation/
│   │   ├── analyze_errors.py
│   │   └── compare_models.py
│   ├── inference/
│   │   └── predict_single.py   # ÚNICO script de predição
│   ├── ocr/
│   │   ├── benchmark_ocrs.py   # Benchmark principal
│   │   ├── exemplo_uso.py      # ÚNICO exemplo
│   │   └── diagnose_enhanced_parseq.py  # ÚNICO diagnóstico
│   ├── setup/
│   │   └── install_ocr_engines.py
│   └── training/
│       └── train_yolo.py        # Script principal de treino
│
├── tests/                       # Testes unitários
│   ├── test_cuda.py
│   ├── test_ocr_module.py
│   └── test_preprocessing.py
│
├── data/                        # Dados (não versionar grandes arquivos)
├── models/                      # Modelos treinados (não versionar)
├── outputs/                     # Resultados (não versionar)
└── logs/                        # Logs (não versionar)
```

---

## 🚀 SCRIPTS PARA AUTOMAÇÃO

Criar scripts auxiliares para facilitar limpeza:

### `cleanup.py` - Deleta arquivos vazios e duplicados automaticamente
### `organize_tests.py` - Move scripts de teste para pasta `tests/`
### `check_imports.py` - Verifica imports quebrados

---

## 📝 OBSERVAÇÕES FINAIS

### O que NÃO deletar:
- ✅ Scripts funcionais de training, evaluation, inference
- ✅ Todo código em `src/` (núcleo do sistema)
- ✅ Configs YAML que são referenciados no código
- ✅ Documentação técnica importante (ARCHITECTURE.md, API.md)

### O que manter bem documentado:
- ✅ Como rodar experimentos principais
- ✅ Como usar a API
- ✅ Como treinar novos modelos
- ✅ Resultados principais do TCC

### Dica para o professor:
Criar um arquivo `QUICKSTART.md` com:
1. Como instalar (3 comandos)
2. Como testar a API (1 exemplo)
3. Como rodar inferência (1 exemplo)
4. Onde estão os resultados principais

---

## ⚠️ ATENÇÃO

**Antes de deletar qualquer arquivo:**
1. ✅ Fazer backup do projeto completo
2. ✅ Commit no git do estado atual
3. ✅ Criar uma branch `cleanup` para as mudanças
4. ✅ Testar que tudo funciona após cada fase

**Comando para criar branch de backup:**
```bash
git checkout -b backup-pre-cleanup
git add .
git commit -m "Backup antes da limpeza"
git checkout -b cleanup
```

---

**Tempo estimado total: 3-4 horas**
**Benefício: Projeto 50% mais limpo e profissional**
