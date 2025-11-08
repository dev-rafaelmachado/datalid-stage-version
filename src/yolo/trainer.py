"""
🏋️ Trainer YOLO
Sistema de treinamento com monitoramento e métricas.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import yaml
from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
    from ultralytics import YOLO
    from ultralytics.utils.callbacks import add_integration_callbacks
except ImportError:
    logger.error(
        "Ultralytics ou TensorBoard não instalado. Instale com: pip install ultralytics tensorboard")
    raise

from ..core.config import config
from ..core.exceptions import (DatasetNotFoundError, GPUNotAvailableError,
                               InsufficientMemoryError, ModelNotFoundError,
                               TrainingError)
from .config import TrainingConfig, YOLOConfig
from .utils import create_data_yaml, optimize_batch_size, validate_gpu


@dataclass
class TrainingMetrics:
    """Métricas de treinamento."""

    # Informações gerais
    model_name: str = ""
    dataset_path: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_epochs: int = 0
    completed_epochs: int = 0

    # Métricas por época
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    map50: List[float] = field(default_factory=list)  # mAP@0.5
    map50_95: List[float] = field(default_factory=list)  # mAP@0.5:0.95
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)

    # Melhor modelo
    best_epoch: int = 0
    best_map50: float = 0.0
    best_map50_95: float = 0.0

    # Métricas do conjunto de teste
    test_results: Optional[Dict[str, float]] = None

    # Hardware
    gpu_memory_used: List[float] = field(default_factory=list)
    training_speed: float = 0.0  # imagens/segundo

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def duration(self) -> Optional[timedelta]:
        """Duração do treinamento."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None

    @property
    def progress(self) -> float:
        """Progresso do treinamento (0-1)."""
        if self.total_epochs > 0:
            return min(self.completed_epochs / self.total_epochs, 1.0)
        return 0.0

    @property
    def eta(self) -> Optional[timedelta]:
        """Tempo estimado para conclusão."""
        if self.progress > 0 and self.duration:
            total_estimated = self.duration / self.progress
            return total_estimated - self.duration
        return None

    def update_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Atualiza métricas da época."""
        self.completed_epochs = epoch + 1

        # Adicionar métricas
        self.train_losses.append(metrics.get('train/loss', 0.0))
        self.val_losses.append(metrics.get('val/loss', 0.0))
        self.map50.append(metrics.get('metrics/mAP50(B)', 0.0))
        self.map50_95.append(metrics.get('metrics/mAP50-95(B)', 0.0))
        self.precision.append(metrics.get('metrics/precision(B)', 0.0))
        self.recall.append(metrics.get('metrics/recall(B)', 0.0))

        # Atualizar melhor modelo
        current_map50 = metrics.get('metrics/mAP50(B)', 0.0)
        if current_map50 > self.best_map50:
            self.best_epoch = epoch
            self.best_map50 = current_map50
            self.best_map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            self.gpu_memory_used.append(gpu_mem)

    def save(self, path: Union[str, Path]) -> None:
        """Salva métricas em arquivo JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Converter para JSON serializable
        data = {
            'model_name': self.model_name,
            'dataset_path': self.dataset_path,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_epochs': self.total_epochs,
            'completed_epochs': self.completed_epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'map50': self.map50,
            'map50_95': self.map50_95,
            'precision': self.precision,
            'recall': self.recall,
            'best_epoch': self.best_epoch,
            'best_map50': self.best_map50,
            'best_map50_95': self.best_map50_95,
            'test_results': self.test_results,
            'gpu_memory_used': self.gpu_memory_used,
            'training_speed': self.training_speed,
            'duration_seconds': self.duration.total_seconds() if self.duration else None
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingMetrics":
        """Carrega métricas de arquivo JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Converter timestamps
        if data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])

        return cls(**{k: v for k, v in data.items() if k != 'duration_seconds'})


class YOLOTrainer:
    """Trainer principal para modelos YOLO."""

    def __init__(
        self,
        config_obj: Optional[YOLOConfig] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Args:
            config_obj: Configuração YOLO
            callbacks: Callbacks customizados
        """
        self.config = config_obj or YOLOConfig()
        self.callbacks = callbacks or []
        self.model = None
        self.metrics = None
        self.is_training = False

        # Validar hardware
        self._validate_hardware()

    def _validate_hardware(self) -> None:
        """Valida hardware disponível."""
        device = str(self.config.training.device)

        if device != 'cpu':
            if not torch.cuda.is_available():
                logger.warning("CUDA não disponível, usando CPU")
                self.config.training.device = 'cpu'
            else:
                # Otimizar batch size para GPU disponível
                optimized_batch = optimize_batch_size(
                    model_name=self.config.training.model,
                    current_batch=self.config.training.batch,
                    device=device
                )

                if optimized_batch != self.config.training.batch:
                    logger.info(
                        f"📊 Batch size otimizado: {self.config.training.batch} → {optimized_batch}")
                    self.config.training.batch = optimized_batch

    def prepare_dataset(
        self,
        data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Prepara dataset para treinamento."""
        data_path = Path(data_path)

        # Verificar se já existe data.yaml
        data_yaml = data_path / 'data.yaml'

        if data_yaml.exists():
            logger.info(f"✅ Dataset já preparado: {data_yaml}")
            return data_yaml

        # Criar data.yaml se necessário
        if output_path:
            output_path = Path(output_path)
            data_yaml = create_data_yaml(
                data_path=data_path,
                output_path=output_path,
                task_type=self.config.training.task_type
            )
        else:
            raise DatasetNotFoundError(
                f"data.yaml não encontrado em {data_path}")

        return data_yaml

    def train(
        self,
        data_path: Union[str, Path],
        resume: bool = False,
        test_after_training: bool = True,
        **overrides
    ) -> TrainingMetrics:
        """
        Executa treinamento.

        Args:
            data_path: Caminho do dataset (pasta com data.yaml)
            resume: Continuar treinamento anterior
            test_after_training: Executar teste no conjunto de teste após treino
            **overrides: Sobrescrever configurações

        Returns:
            Métricas do treinamento
        """
        try:
            logger.info("🚀 Iniciando treinamento YOLO")
            self._log_training_info()

            # Preparar dataset
            data_yaml = self.prepare_dataset(data_path)
            self.config.training.data = str(data_yaml)

            # Aplicar overrides
            for key, value in overrides.items():
                if hasattr(self.config.training, key):
                    setattr(self.config.training, key, value)

            # Carregar modelo
            self.model = YOLO(self.config.training.model)

            # Configurar métricas
            self.metrics = TrainingMetrics(
                model_name=self.config.training.model,
                dataset_path=str(data_path),
                total_epochs=self.config.training.epochs
            )

            # Configurar callbacks
            self._setup_callbacks()

            # Argumentos de treinamento
            train_args = self.config.training.to_ultralytics_args()

            logger.info("🏋️ Iniciando treinamento...")
            self.is_training = True

            # --- INÍCIO INTEGRAÇÃO TENSORBOARD ---
            log_dir = Path(train_args['project']) / \
                train_args.get('name', 'exp') / 'tensorboard_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(log_dir))
            logger.success(f"📊 TensorBoard logs serão salvos em: {log_dir}")
            logger.info(
                f"🔍 Para visualizar em tempo real: tensorboard --logdir=experiments")
            # --- FIM INÍCIO INTEGRAÇÃO TENSORBOARD ---

            # Treinar modelo
            def on_train_epoch_end_tensorboard(trainer):
                """Callback para logar métricas no TensorBoard em tempo real."""
                try:
                    epoch = trainer.epoch
                    logger.info(
                        f"📊 Callback TensorBoard chamado para época {epoch}")

                    # Capturar métricas do trainer
                    metrics_dict = {}
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        metrics_dict = dict(trainer.metrics)
                    elif hasattr(trainer, 'validator') and trainer.validator:
                        if hasattr(trainer.validator, 'metrics'):
                            metrics_dict = dict(trainer.validator.metrics)

                    # Também tentar pegar do CSV do trainer
                    if hasattr(trainer, 'csv') and trainer.csv:
                        csv_data = trainer.csv
                        if isinstance(csv_data, dict):
                            metrics_dict.update(csv_data)

                    logger.info(
                        f"📊 Métricas disponíveis: {list(metrics_dict.keys())}")

                    # Atualizar métricas internas
                    if self.metrics:
                        self.metrics.update_epoch(epoch, metrics_dict)

                    # --- LOG TENSORBOARD ---
                    # Todas as métricas do CSV
                    metric_names = [
                        'train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss',
                        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                        'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)',
                        'val/box_loss', 'val/seg_loss', 'val/cls_loss', 'val/dfl_loss',
                        'lr/pg0', 'lr/pg1', 'lr/pg2', 'epoch', 'time'
                    ]

                    logged_count = 0
                    for name in metric_names:
                        if name in metrics_dict and metrics_dict[name] is not None:
                            try:
                                value = float(metrics_dict[name])
                                tb_writer.add_scalar(name, value, epoch)
                                logged_count += 1
                                logger.debug(f"  ✓ {name}: {value}")
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"⚠️ Erro ao logar métrica {name}: {e}")

                    # Nome do experimento
                    if epoch == 0:
                        tb_writer.add_text(
                            'Config/experiment_name', train_args.get('name', 'exp'), epoch)

                    # IMPORTANTE: Fazer flush para atualização em tempo real
                    tb_writer.flush()

                    logger.success(
                        f"✅ Época {epoch}: {logged_count} métricas logadas no TensorBoard")

                except Exception as e:
                    logger.error(f"❌ Erro no callback do TensorBoard: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # Adicionar callback para TensorBoard ANTES de treinar
            if hasattr(self.model, 'add_callback'):
                self.model.add_callback(
                    'on_train_epoch_end', on_train_epoch_end_tensorboard)
                logger.success(
                    "✅ Callback do TensorBoard registrado com sucesso!")
            else:
                logger.warning(
                    "⚠️ Modelo não suporta callbacks - TensorBoard não será atualizado em tempo real")
            # --- FIM INTEGRAÇÃO TENSORBOARD ---

            results = self.model.train(**train_args)

            # Finalizar métricas
            self.metrics.end_time = datetime.now()
            self.is_training = False
            # Fechar writer do TensorBoard
            tb_writer.close()
            logger.success("✅ Treinamento concluído!")
            self._log_training_results()

            # Executar teste no conjunto de teste
            if test_after_training:
                logger.info("=" * 70)
                logger.info("🧪 Executando avaliação final no conjunto de TESTE...")
                logger.info("=" * 70)
                try:
                    test_metrics = self.test_model(data_path)
                    self.metrics.test_results = test_metrics
                    self._log_test_results(test_metrics)
                    
                    # Salvar resultados do teste separadamente
                    test_results_path = Path(
                        train_args['project']) / train_args.get('name', 'exp') / 'test_results.json'
                    with open(test_results_path, 'w', encoding='utf-8') as f:
                        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
                    logger.info(f"📄 Resultados do teste salvos em: {test_results_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao executar teste: {str(e)}")
                    logger.warning("Continuando sem métricas de teste...")

            # Salvar métricas
            metrics_path = Path(
                train_args['project']) / train_args.get('name', 'exp') / 'metrics.json'
            self.metrics.save(metrics_path)

            return self.metrics

        except Exception as e:
            self.is_training = False
            logger.error(f"❌ Erro no treinamento: {str(e)}")
            raise TrainingError(f"Erro no treinamento: {str(e)}")

    def _setup_callbacks(self) -> None:
        """Configura callbacks de treinamento."""
        def on_train_epoch_end(trainer):
            """Callback executado ao final de cada época."""
            if self.metrics:
                # Extrair métricas do trainer
                metrics_dict = {}
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    metrics_dict = trainer.metrics
                elif hasattr(trainer, 'validator') and trainer.validator:
                    if hasattr(trainer.validator, 'metrics'):
                        metrics_dict = trainer.validator.metrics

                self.metrics.update_epoch(trainer.epoch, metrics_dict)

                # Log progresso
                progress = self.metrics.progress * 100
                eta_str = f"ETA: {self.metrics.eta}" if self.metrics.eta else "ETA: N/A"

                logger.info(f"📊 Época {trainer.epoch + 1}/{self.metrics.total_epochs} "
                            f"({progress:.1f}%) - {eta_str}")

                if self.metrics.map50:
                    logger.info(f"🎯 mAP50: {self.metrics.map50[-1]:.3f} "
                                f"(melhor: {self.metrics.best_map50:.3f} @ época {self.metrics.best_epoch + 1})")

        # Adicionar callback personalizado
        if hasattr(self.model, 'add_callback'):
            self.model.add_callback('on_train_epoch_end', on_train_epoch_end)

    def _log_training_info(self) -> None:
        """Log informações do treinamento."""
        tc = self.config.training

        logger.info("📋 CONFIGURAÇÃO DE TREINAMENTO:")
        logger.info(f"  • Modelo: {tc.model}")
        logger.info(f"  • Tarefa: {tc.task_type}")
        logger.info(f"  • Épocas: {tc.epochs}")
        logger.info(f"  • Batch size: {tc.batch}")
        logger.info(f"  • Imagem: {tc.imgsz}px")
        logger.info(f"  • Dispositivo: {tc.device}")
        logger.info(f"  • Workers: {tc.workers}")
        logger.info(
            f"  • Augmentations: {'✅' if tc.augmentation.enabled else '❌'}")

        # Estimar tempo
        if tc.data:
            try:
                with open(tc.data, 'r') as f:
                    data_config = yaml.safe_load(f)

                # Contar imagens de treino (estimativa)
                train_path = Path(data_config['path']) / data_config['train']
                if train_path.exists():
                    train_images = len(list(train_path.glob(
                        '*.jpg'))) + len(list(train_path.glob('*.png')))
                    time_estimate = tc.estimate_training_time(train_images)
                    logger.info(
                        f"⏱️ Tempo estimado: {time_estimate['estimated_completion']}")
            except:
                pass

    def _log_training_results(self) -> None:
        """Log resultados do treinamento."""
        if not self.metrics:
            return

        logger.info("🎉 RESULTADOS DO TREINAMENTO:")
        logger.info(f"  • Duração: {self.metrics.duration}")
        logger.info(
            f"  • Épocas completadas: {self.metrics.completed_epochs}/{self.metrics.total_epochs}")
        logger.info(
            f"  • Melhor mAP50: {self.metrics.best_map50:.3f} (época {self.metrics.best_epoch + 1})")
        logger.info(f"  • mAP50-95: {self.metrics.best_map50_95:.3f}")

        if self.metrics.gpu_memory_used:
            max_gpu_mem = max(self.metrics.gpu_memory_used)
            logger.info(f"  • GPU Memory pico: {max_gpu_mem:.1f}GB")

    def test_model(
        self,
        data_path: Union[str, Path],
        split: str = 'test'
    ) -> Dict[str, float]:
        """
        Testa modelo no conjunto de teste.
        
        Args:
            data_path: Caminho do dataset
            split: Split para testar ('test' por padrão)
            
        Returns:
            Dicionário com métricas do teste
        """
        if not self.model:
            raise ModelNotFoundError("Modelo não treinado")

        logger.info(f"🧪 Testando modelo no split '{split}'...")

        # Preparar dataset
        data_yaml = self.prepare_dataset(data_path)

        # Executar validação no split de teste
        # Tentar com save_json, se falhar (faster_coco_eval não disponível), usar sem
        try:
            results = self.model.val(
                data=str(data_yaml),
                split=split,
                plots=True,
                save_json=True,
                verbose=False
            )
        except ModuleNotFoundError as e:
            if 'faster_coco_eval' in str(e):
                logger.warning("⚠️ faster_coco_eval não disponível, continuando sem save_json")
                logger.info("💡 Para habilitar: pip install faster-coco-eval")
                results = self.model.val(
                    data=str(data_yaml),
                    split=split,
                    plots=True,
                    save_json=False,
                    verbose=False
                )
            else:
                raise

        # Extrair métricas
        test_metrics = {
            'split': split,
        }

        # Métricas de detecção (box)
        if hasattr(results, 'box') and results.box is not None:
            test_metrics.update({
                'box_mAP50': float(results.box.map50),
                'box_mAP50_95': float(results.box.map),
                'box_mAP75': float(results.box.map75) if hasattr(results.box, 'map75') else None,
                'box_precision': float(results.box.p),
                'box_recall': float(results.box.r),
            })
            
            # Calcular F1
            p, r = results.box.p, results.box.r
            if p > 0 and r > 0:
                test_metrics['box_f1'] = float(2 * p * r / (p + r))
            else:
                test_metrics['box_f1'] = 0.0

        # Métricas de segmentação (mask)
        if hasattr(results, 'masks') and results.masks is not None:
            test_metrics.update({
                'mask_mAP50': float(results.masks.map50),
                'mask_mAP50_95': float(results.masks.map),
                'mask_mAP75': float(results.masks.map75) if hasattr(results.masks, 'map75') else None,
                'mask_precision': float(results.masks.p),
                'mask_recall': float(results.masks.r),
            })
            
            # Calcular F1
            p, r = results.masks.p, results.masks.r
            if p > 0 and r > 0:
                test_metrics['mask_f1'] = float(2 * p * r / (p + r))
            else:
                test_metrics['mask_f1'] = 0.0

        logger.success(f"✅ Teste no split '{split}' concluído!")
        return test_metrics

    def _log_test_results(self, test_metrics: Dict[str, float]) -> None:
        """
        Log resultados do teste.
        """
        logger.info("=" * 70)
        logger.info(f"🎯 MÉTRICAS DO CONJUNTO DE {test_metrics.get('split', 'TEST').upper()}")
        logger.info("=" * 70)
        
        # Box metrics
        if 'box_mAP50' in test_metrics:
            logger.info("📦 Métricas de Detecção (Box):")
            logger.info(f"  • mAP@0.5      : {test_metrics['box_mAP50']:.4f}")
            logger.info(f"  • mAP@0.5:0.95 : {test_metrics['box_mAP50_95']:.4f}")
            if test_metrics.get('box_mAP75') is not None:
                logger.info(f"  • mAP@0.75     : {test_metrics['box_mAP75']:.4f}")
            logger.info(f"  • Precision    : {test_metrics['box_precision']:.4f}")
            logger.info(f"  • Recall       : {test_metrics['box_recall']:.4f}")
            logger.info(f"  • F1-Score     : {test_metrics['box_f1']:.4f}")
        
        # Mask metrics
        if 'mask_mAP50' in test_metrics:
            logger.info("")
            logger.info("🎭 Métricas de Segmentação (Mask):")
            logger.info(f"  • mAP@0.5      : {test_metrics['mask_mAP50']:.4f}")
            logger.info(f"  • mAP@0.5:0.95 : {test_metrics['mask_mAP50_95']:.4f}")
            if test_metrics.get('mask_mAP75') is not None:
                logger.info(f"  • mAP@0.75     : {test_metrics['mask_mAP75']:.4f}")
            logger.info(f"  • Precision    : {test_metrics['mask_precision']:.4f}")
            logger.info(f"  • Recall       : {test_metrics['mask_recall']:.4f}")
            logger.info(f"  • F1-Score     : {test_metrics['mask_f1']:.4f}")
        
        logger.info("=" * 70)

    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        **overrides
    ) -> TrainingMetrics:
        """Resume treinamento de checkpoint."""
        logger.info(f"🔄 Resumindo treinamento: {checkpoint_path}")

        # Carregar modelo do checkpoint
        self.model = YOLO(checkpoint_path)

        # Continuar treinamento
        return self.train(resume=True, **overrides)

    def validate(
        self,
        data_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Valida modelo treinado."""
        if not self.model:
            raise ModelNotFoundError("Modelo não treinado")

        logger.info("🔍 Validando modelo...")

        if data_path:
            data_yaml = self.prepare_dataset(data_path)
            kwargs['data'] = str(data_yaml)

        results = self.model.val(**kwargs)

        logger.success("✅ Validação concluída")
        return results.results_dict if hasattr(results, 'results_dict') else {}

    def export_model(
        self,
        format: str = 'onnx',
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Path:
        """Exporta modelo treinado."""
        if not self.model:
            raise ModelNotFoundError("Modelo não treinado")

        logger.info(f"📦 Exportando modelo para {format.upper()}...")

        exported_path = self.model.export(format=format, **kwargs)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            exported_path = Path(exported_path)
            final_path = output_dir / exported_path.name
            exported_path.rename(final_path)
            exported_path = final_path

        logger.success(f"✅ Modelo exportado: {exported_path}")
        return Path(exported_path)
