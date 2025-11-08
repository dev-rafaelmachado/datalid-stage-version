#!/usr/bin/env python3
"""
🧪 Script de teste rápido para validar avaliação automática

Testa a funcionalidade de avaliação automática no conjunto de teste.
"""

import sys
from pathlib import Path

# Adicionar src ao path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

from src.yolo import TrainingConfig, YOLOConfig, YOLOTrainer


def test_automatic_evaluation():
    """Testa avaliação automática após treinamento."""
    logger.info("🧪 Testando avaliação automática no conjunto de teste")
    logger.info("=" * 70)
    
    # Configurar para treinamento rápido (apenas para teste)
    config = YOLOConfig()
    config.training = TrainingConfig(
        model='yolov8n-seg.pt',  # Modelo pequeno
        epochs=2,  # Apenas 2 épocas para teste rápido
        batch=4,
        imgsz=320,  # Imagem menor para velocidade
        patience=50,
        workers=2
    )
    
    # Criar trainer
    trainer = YOLOTrainer(config_obj=config)
    
    # Dataset
    data_path = ROOT_DIR / 'data' / 'processed' / 'v1_segment'
    
    if not data_path.exists():
        logger.error(f"❌ Dataset não encontrado: {data_path}")
        logger.info("💡 Execute prepare_dataset.py primeiro")
        return False
    
    try:
        logger.info("🏋️ Iniciando treinamento de teste (2 épocas)...")
        
        # Treinar COM teste automático
        metrics = trainer.train(
            data_path=data_path,
            test_after_training=True,  # Ativar teste automático
            project='experiments',
            name='test_auto_eval'
        )
        
        # Verificar se teste foi executado
        if metrics.test_results:
            logger.success("✅ Teste automático executado com sucesso!")
            logger.info(f"📊 Métricas de teste: {metrics.test_results}")
            
            # Verificar arquivo test_results.json
            test_results_path = ROOT_DIR / 'experiments' / 'test_auto_eval' / 'test_results.json'
            if test_results_path.exists():
                logger.success(f"✅ Arquivo test_results.json criado: {test_results_path}")
            else:
                logger.warning("⚠️ Arquivo test_results.json não encontrado")
            
            return True
        else:
            logger.error("❌ Teste automático não foi executado")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_disable_automatic_evaluation():
    """Testa desabilitar avaliação automática."""
    logger.info("\n🧪 Testando DESABILITAR avaliação automática")
    logger.info("=" * 70)
    
    # Configurar para treinamento rápido
    config = YOLOConfig()
    config.training = TrainingConfig(
        model='yolov8n-seg.pt',
        epochs=1,
        batch=4,
        imgsz=320,
        patience=50,
        workers=2
    )
    
    trainer = YOLOTrainer(config_obj=config)
    data_path = ROOT_DIR / 'data' / 'processed' / 'v1_segment'
    
    if not data_path.exists():
        logger.error(f"❌ Dataset não encontrado: {data_path}")
        return False
    
    try:
        logger.info("🏋️ Iniciando treinamento sem teste automático...")
        
        # Treinar SEM teste automático
        metrics = trainer.train(
            data_path=data_path,
            test_after_training=False,  # Desabilitar teste automático
            project='experiments',
            name='test_no_auto_eval'
        )
        
        # Verificar que teste NÃO foi executado
        if metrics.test_results is None:
            logger.success("✅ Teste automático corretamente desabilitado!")
            return True
        else:
            logger.error("❌ Teste foi executado mesmo com test_after_training=False")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro durante teste: {str(e)}")
        return False


def main():
    """Função principal."""
    logger.info("🚀 TESTE DE AVALIAÇÃO AUTOMÁTICA")
    logger.info("=" * 70)
    logger.info("Este script testa a funcionalidade de avaliação automática")
    logger.info("Será executado um treinamento curto (2 épocas) para validar")
    logger.info("=" * 70)
    logger.info("")
    
    # Teste 1: Com avaliação automática
    test1_passed = test_automatic_evaluation()
    
    # Teste 2: Sem avaliação automática
    test2_passed = test_disable_automatic_evaluation()
    
    # Resultado final
    logger.info("\n" + "=" * 70)
    logger.info("📊 RESULTADOS DOS TESTES")
    logger.info("=" * 70)
    logger.info(f"Teste 1 (com avaliação): {'✅ PASSOU' if test1_passed else '❌ FALHOU'}")
    logger.info(f"Teste 2 (sem avaliação): {'✅ PASSOU' if test2_passed else '❌ FALHOU'}")
    logger.info("=" * 70)
    
    if test1_passed and test2_passed:
        logger.success("🎉 TODOS OS TESTES PASSARAM!")
        return 0
    else:
        logger.error("❌ ALGUNS TESTES FALHARAM")
        return 1


if __name__ == "__main__":
    sys.exit(main())
