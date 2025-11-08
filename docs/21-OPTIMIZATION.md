# 21. Performance Optimization

## Table of Contents

1. [Overview](#overview)
2. [GPU Optimization](#gpu-optimization)
3. [Batch Processing](#batch-processing)
4. [Memory Management](#memory-management)
5. [Caching Strategies](#caching-strategies)
6. [Model Optimization](#model-optimization)
7. [Preprocessing Optimization](#preprocessing-optimization)
8. [Distributed Processing](#distributed-processing)
9. [Profiling and Benchmarking](#profiling-and-benchmarking)
10. [Production Best Practices](#production-best-practices)

---

## Overview

This guide covers comprehensive performance optimization strategies for Datalid 3.0, from development to production deployment. Learn how to maximize throughput, minimize latency, and efficiently utilize resources.

### Performance Targets

| Metric | Development | Production | High-Performance |
|--------|-------------|------------|------------------|
| **Throughput** | 5-10 img/s | 20-50 img/s | 100+ img/s |
| **Latency** | < 1s | < 500ms | < 100ms |
| **Memory** | < 4GB | < 8GB | Optimized |
| **GPU Utilization** | 50-70% | 80-95% | 95%+ |

### Quick Wins

```python
# Enable these for immediate improvements
optimization_config = {
    'gpu': True,                    # +500% speed
    'batch_size': 16,              # +200% throughput
    'half_precision': True,        # +50% speed, -50% memory
    'preprocessing_cache': True,   # +100% on repeated images
    'async_processing': True,      # Better resource utilization
}
```

---

## GPU Optimization

### 1. GPU Selection and Setup

#### Check GPU Availability

```python
import torch
import tensorflow as tf

def check_gpu_setup():
    """Verify GPU configuration"""
    print("=== PyTorch GPU Setup ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n=== TensorFlow GPU Setup ===")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    return torch.cuda.is_available()

# Run check
gpu_available = check_gpu_setup()
```

#### Configure GPU Memory

```python
# PyTorch - Set memory fraction
import torch

def configure_pytorch_gpu(memory_fraction=0.9):
    """Configure PyTorch GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.cuda.empty_cache()
        print(f"GPU memory limit set to {memory_fraction*100}%")

# TensorFlow - Dynamic memory growth
import tensorflow as tf

def configure_tensorflow_gpu():
    """Configure TensorFlow GPU memory usage"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Or set memory limit
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
            )
            print("TensorFlow GPU configured")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

# Apply configurations
configure_pytorch_gpu(0.9)
configure_tensorflow_gpu()
```

### 2. Mixed Precision Training/Inference

#### FP16 (Half Precision)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class OptimizedYOLOInference:
    """YOLO inference with FP16 optimization"""
    
    def __init__(self, model_path, use_fp16=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        if self.use_fp16:
            self.model.half()
            print("Using FP16 inference")
    
    def predict(self, image):
        """Run inference with automatic mixed precision"""
        with torch.no_grad():
            if self.use_fp16:
                with autocast():
                    results = self.model(image)
            else:
                results = self.model(image)
        return results

# Usage
model = OptimizedYOLOInference('yolov8n-seg.pt', use_fp16=True)
```

#### Configuration for Mixed Precision

```yaml
# config/optimization/mixed_precision.yaml
yolo:
  use_amp: true              # Automatic Mixed Precision
  amp_dtype: float16         # float16 or bfloat16
  
ocr:
  parseq:
    use_fp16: true
    device: cuda
  
  trocr:
    use_fp16: true
    torch_dtype: float16
  
  paddleocr:
    use_fp16: true
```

### 3. CUDA Optimization

```python
import torch

def optimize_cuda_settings():
    """Apply CUDA optimizations"""
    if torch.cuda.is_available():
        # Enable TF32 (Tensor Float 32) on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Disable cuDNN deterministic (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        
        print("CUDA optimizations enabled")

optimize_cuda_settings()
```

### 4. Multi-GPU Processing

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

class MultiGPUProcessor:
    """Process images across multiple GPUs"""
    
    def __init__(self, model_path, gpu_ids=[0, 1]):
        self.device = torch.device('cuda:0')
        self.model = torch.load(model_path)
        
        # DataParallel for simple multi-GPU
        if len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids)
            print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def process_batch(self, images):
        """Process batch across GPUs"""
        with torch.no_grad():
            images = images.to(self.device)
            results = self.model(images)
        return results

# Usage
processor = MultiGPUProcessor('yolov8n-seg.pt', gpu_ids=[0, 1, 2, 3])
```

---

## Batch Processing

### 1. Dynamic Batching

```python
import numpy as np
from collections import deque
import time

class DynamicBatcher:
    """Dynamically batch requests for optimal throughput"""
    
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.results = {}
    
    def add_request(self, request_id, image):
        """Add request to batch queue"""
        self.queue.append((request_id, image, time.time()))
    
    def should_process_batch(self):
        """Determine if batch should be processed"""
        if not self.queue:
            return False
        
        # Process if batch is full
        if len(self.queue) >= self.max_batch_size:
            return True
        
        # Process if oldest request has waited too long
        oldest_time = self.queue[0][2]
        if time.time() - oldest_time >= self.max_wait_time:
            return True
        
        return False
    
    def get_batch(self):
        """Extract batch from queue"""
        batch_size = min(len(self.queue), self.max_batch_size)
        batch = []
        request_ids = []
        
        for _ in range(batch_size):
            req_id, image, _ = self.queue.popleft()
            batch.append(image)
            request_ids.append(req_id)
        
        return request_ids, np.array(batch)
    
    def process_loop(self, model):
        """Main processing loop"""
        while True:
            if self.should_process_batch():
                request_ids, batch = self.get_batch()
                
                # Process batch
                results = model.predict(batch)
                
                # Store results
                for req_id, result in zip(request_ids, results):
                    self.results[req_id] = result
            else:
                time.sleep(0.001)  # Small sleep to avoid busy waiting

# Usage
batcher = DynamicBatcher(max_batch_size=16, max_wait_time=0.05)
```

### 2. Batch Pipeline Configuration

```yaml
# config/optimization/batch_processing.yaml
pipeline:
  batch_processing:
    enabled: true
    
    # YOLO detection batching
    yolo:
      batch_size: 16
      max_wait_ms: 50
      queue_size: 100
      timeout_ms: 5000
    
    # OCR batching
    ocr:
      batch_size: 32
      max_wait_ms: 100
      queue_size: 200
      timeout_ms: 10000
    
    # Preprocessing batching
    preprocessing:
      parallel_workers: 4
      queue_size: 50
```

### 3. Optimal Batch Sizes

```python
def find_optimal_batch_size(model, image_shape, max_memory_gb=8):
    """Find optimal batch size for GPU memory"""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_memory = max_memory_gb * 1e9
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    optimal_batch = 1
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create dummy batch
            dummy_input = torch.randn(batch_size, *image_shape).to(device)
            
            # Test inference
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated()
                if memory_used < max_memory:
                    optimal_batch = batch_size
                    print(f"Batch {batch_size}: {memory_used/1e9:.2f} GB ✓")
                else:
                    print(f"Batch {batch_size}: OOM")
                    break
            else:
                optimal_batch = batch_size
                
        except RuntimeError as e:
            print(f"Batch {batch_size}: Failed - {e}")
            break
    
    print(f"\nOptimal batch size: {optimal_batch}")
    return optimal_batch

# Usage
optimal_batch = find_optimal_batch_size(
    model=yolo_model,
    image_shape=(3, 640, 640),
    max_memory_gb=8
)
```

---

## Memory Management

### 1. Memory Profiling

```python
import psutil
import torch
import tracemalloc

class MemoryProfiler:
    """Profile memory usage"""
    
    def __init__(self):
        self.start_memory = None
        tracemalloc.start()
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        self.start_memory = psutil.Process().memory_info().rss / 1e9
        return self
    
    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_memory = psutil.Process().memory_info().rss / 1e9
        ram_used = end_memory - self.start_memory
        
        print("\n=== Memory Profile ===")
        print(f"RAM Used: {ram_used:.2f} GB")
        print(f"Python Peak: {peak/1e9:.2f} GB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU Used: {gpu_memory:.2f} GB")

# Usage
with MemoryProfiler():
    results = process_images(image_batch)
```

### 2. Memory Optimization Techniques

```python
class MemoryOptimizedPipeline:
    """Pipeline with memory optimizations"""
    
    def __init__(self):
        self.yolo = None
        self.ocr = None
    
    def lazy_load_models(self):
        """Load models only when needed"""
        if self.yolo is None:
            from src.yolo import YOLODetector
            self.yolo = YOLODetector('yolov8n-seg.pt')
        
        if self.ocr is None:
            from src.ocr import OCREngine
            self.ocr = OCREngine('parseq')
    
    def process_with_cleanup(self, image):
        """Process with aggressive memory cleanup"""
        import gc
        
        # Detection
        detections = self.yolo.detect(image)
        
        # Free detection intermediates
        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # OCR on crops
        results = []
        for detection in detections:
            crop = detection['crop']
            text = self.ocr.recognize(crop)
            results.append(text)
            
            # Free crop
            del crop
        
        # Final cleanup
        del detections
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def __del__(self):
        """Cleanup on deletion"""
        import gc
        self.yolo = None
        self.ocr = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 3. Memory-Efficient Data Loading

```python
from torch.utils.data import Dataset, DataLoader
import cv2

class StreamingDataset(Dataset):
    """Memory-efficient dataset that loads images on-demand"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image only when accessed
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Usage with DataLoader
dataset = StreamingDataset(image_paths)
loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)

for batch in loader:
    results = model.predict(batch)
```

---

## Caching Strategies

### 1. Result Caching

```python
import hashlib
import pickle
from pathlib import Path
import redis

class ResultCache:
    """Cache processing results"""
    
    def __init__(self, cache_dir='cache/', use_redis=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_redis = use_redis
        
        if use_redis:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def _get_hash(self, image):
        """Generate hash for image"""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def get(self, image):
        """Get cached result"""
        key = self._get_hash(image)
        
        if self.use_redis:
            result = self.redis_client.get(key)
            if result:
                return pickle.loads(result)
        else:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def set(self, image, result, ttl=3600):
        """Cache result"""
        key = self._get_hash(image)
        serialized = pickle.dumps(result)
        
        if self.use_redis:
            self.redis_client.setex(key, ttl, serialized)
        else:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
    
    def clear(self):
        """Clear cache"""
        if self.use_redis:
            self.redis_client.flushdb()
        else:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()

# Usage
cache = ResultCache(use_redis=True)

def process_with_cache(image):
    """Process with caching"""
    # Check cache
    result = cache.get(image)
    if result:
        print("Cache hit!")
        return result
    
    # Process
    result = model.predict(image)
    
    # Cache result
    cache.set(image, result, ttl=3600)
    
    return result
```

### 2. Model Caching

```python
from functools import lru_cache

class ModelManager:
    """Manage model loading with caching"""
    
    def __init__(self):
        self._models = {}
    
    @lru_cache(maxsize=5)
    def get_yolo_model(self, model_name):
        """Get YOLO model (cached)"""
        from ultralytics import YOLO
        return YOLO(f'{model_name}.pt')
    
    @lru_cache(maxsize=10)
    def get_ocr_engine(self, engine_type):
        """Get OCR engine (cached)"""
        from src.ocr.factory import OCRFactory
        return OCRFactory.create_engine(engine_type)
    
    def preload_models(self, yolo_models, ocr_engines):
        """Preload models into cache"""
        for model in yolo_models:
            self.get_yolo_model(model)
        
        for engine in ocr_engines:
            self.get_ocr_engine(engine)
        
        print(f"Preloaded {len(yolo_models)} YOLO models and {len(ocr_engines)} OCR engines")

# Usage
manager = ModelManager()
manager.preload_models(
    yolo_models=['yolov8n-seg', 'yolov8s-seg'],
    ocr_engines=['parseq', 'paddleocr']
)
```

### 3. Preprocessing Cache

```python
import cv2
import numpy as np
from diskcache import Cache

class PreprocessingCache:
    """Cache preprocessed images"""
    
    def __init__(self, cache_dir='preprocessing_cache/', size_limit_gb=10):
        self.cache = Cache(cache_dir, size_limit=size_limit_gb * 1e9)
    
    def get_or_preprocess(self, image_path, preprocessing_func):
        """Get preprocessed image or compute and cache"""
        key = f"{image_path}_{preprocessing_func.__name__}"
        
        # Check cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Preprocess
        image = cv2.imread(image_path)
        processed = preprocessing_func(image)
        
        # Cache
        self.cache.set(key, processed)
        
        return processed

# Usage
prep_cache = PreprocessingCache()

def my_preprocessing(image):
    # Expensive preprocessing
    return processed_image

processed = prep_cache.get_or_preprocess('image.jpg', my_preprocessing)
```

---

## Model Optimization

### 1. Model Quantization

```python
import torch
from torch.quantization import quantize_dynamic

def quantize_model(model, quantization_type='dynamic'):
    """Quantize model for faster inference"""
    
    if quantization_type == 'dynamic':
        # Dynamic quantization (easiest)
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
    
    elif quantization_type == 'static':
        # Static quantization (requires calibration)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        # calibrate(model, calibration_data)
        
        torch.quantization.convert(model, inplace=True)
        quantized_model = model
    
    return quantized_model

# Usage
quantized_yolo = quantize_model(yolo_model, 'dynamic')

# Compare sizes
original_size = sum(p.numel() for p in yolo_model.parameters()) * 4 / 1e6
quantized_size = sum(p.numel() for p in quantized_yolo.parameters()) / 1e6
print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

### 2. Model Pruning

```python
import torch.nn.utils.prune as prune

def prune_model(model, pruning_amount=0.3):
    """Prune model to reduce size and increase speed"""
    
    # Prune convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    
    # Check sparsity
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params * 100
    print(f"Model sparsity: {sparsity:.2f}%")
    
    return model

# Usage
pruned_model = prune_model(model, pruning_amount=0.3)
```

### 3. ONNX Export

```python
import torch
import onnx
import onnxruntime as ort

def export_to_onnx(model, input_shape, output_path='model.onnx'):
    """Export PyTorch model to ONNX"""
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
    
    return output_path

def onnx_inference(onnx_path, input_data):
    """Run inference with ONNX Runtime"""
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: input_data})
    return result[0]

# Usage
onnx_path = export_to_onnx(model, input_shape=(3, 640, 640))
result = onnx_inference(onnx_path, input_data)
```

### 4. TensorRT Optimization

```python
def export_to_tensorrt(onnx_path, engine_path='model.engine', fp16=True):
    """Convert ONNX to TensorRT engine"""
    import tensorrt as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    
    # Serialize
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {engine_path}")
    return engine_path

# Usage
tensorrt_engine = export_to_tensorrt('model.onnx', fp16=True)
```

---

## Preprocessing Optimization

### 1. Parallel Preprocessing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cv2
import numpy as np

class ParallelPreprocessor:
    """Parallel image preprocessing"""
    
    def __init__(self, num_workers=4, use_processes=False):
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(num_workers) if use_processes else ThreadPoolExecutor(num_workers)
    
    def preprocess_single(self, image_path):
        """Preprocess single image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        return image
    
    def preprocess_batch(self, image_paths):
        """Preprocess batch in parallel"""
        results = list(self.executor.map(self.preprocess_single, image_paths))
        return np.array(results)
    
    def __del__(self):
        self.executor.shutdown()

# Usage
preprocessor = ParallelPreprocessor(num_workers=8)
batch = preprocessor.preprocess_batch(image_paths)
```

### 2. GPU-Accelerated Preprocessing

```python
import torch
import torchvision.transforms as T

class GPUPreprocessor:
    """GPU-accelerated preprocessing"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((640, 640)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_batch(self, images):
        """Preprocess batch on GPU"""
        # Convert to tensor
        if not isinstance(images, torch.Tensor):
            images = torch.stack([self.transform(img) for img in images])
        
        # Move to GPU
        images = images.to(self.device)
        
        return images

# Usage
gpu_prep = GPUPreprocessor()
processed = gpu_prep.preprocess_batch(image_batch)
```

### 3. OpenCV Optimization

```python
import cv2

def optimize_opencv():
    """Apply OpenCV optimizations"""
    # Use optimized code paths
    cv2.setUseOptimized(True)
    
    # Set number of threads
    cv2.setNumThreads(4)
    
    # Check optimizations
    print(f"OpenCV Optimized: {cv2.useOptimized()}")
    print(f"OpenCV Threads: {cv2.getNumThreads()}")
    
    # Use hardware acceleration if available
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print("OpenCL enabled")

optimize_opencv()

# Fast resize with INTER_LINEAR
def fast_resize(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

# Fast color conversion
def fast_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

## Distributed Processing

### 1. Ray for Distributed Computing

```python
import ray
from ray import serve

# Initialize Ray
ray.init()

@ray.remote
class DistributedProcessor:
    """Distributed image processor"""
    
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
    
    def process(self, image):
        """Process single image"""
        return self.model.predict(image)

# Create workers
num_workers = 4
workers = [DistributedProcessor.remote('yolov8n-seg.pt') for _ in range(num_workers)]

# Distribute work
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
futures = [workers[i % num_workers].process.remote(path) for i, path in enumerate(image_paths)]

# Collect results
results = ray.get(futures)

# Shutdown
ray.shutdown()
```

### 2. Celery for Task Queue

```python
from celery import Celery

# Initialize Celery
app = Celery('datalid', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

@app.task
def process_image_task(image_path):
    """Celery task for image processing"""
    from src.pipeline import FullPipeline
    
    pipeline = FullPipeline()
    result = pipeline.process(image_path)
    
    return result

# Submit tasks
from celery import group

job = group(process_image_task.s(path) for path in image_paths)
result = job.apply_async()

# Wait for results
outputs = result.get()
```

### 3. Multi-Node Processing with Dask

```python
from dask.distributed import Client, LocalCluster
import dask.bag as db

# Setup cluster
cluster = LocalCluster(n_workers=4, threads_per_worker=2)
client = Client(cluster)

def process_image(image_path):
    """Process function"""
    from src.pipeline import FullPipeline
    pipeline = FullPipeline()
    return pipeline.process(image_path)

# Create Dask bag
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'] * 100
bag = db.from_sequence(image_paths, partition_size=10)

# Process in parallel
results = bag.map(process_image).compute()

# Cleanup
client.close()
cluster.close()
```

---

## Profiling and Benchmarking

### 1. Comprehensive Profiler

```python
import time
import cProfile
import pstats
from functools import wraps
import torch

class PerformanceProfiler:
    """Comprehensive performance profiler"""
    
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def measure(self, name):
        """Decorator to measure function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                elapsed = end - start
                if name not in self.times:
                    self.times[name] = []
                    self.counts[name] = 0
                
                self.times[name].append(elapsed)
                self.counts[name] += 1
                
                return result
            return wrapper
        return decorator
    
    def report(self):
        """Generate performance report"""
        import numpy as np
        
        print("\n=== Performance Report ===")
        print(f"{'Function':<30} {'Calls':<10} {'Total (s)':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 90)
        
        for name in sorted(self.times.keys()):
            times = np.array(self.times[name])
            total = times.sum()
            avg = times.mean() * 1000
            min_time = times.min() * 1000
            max_time = times.max() * 1000
            count = self.counts[name]
            
            print(f"{name:<30} {count:<10} {total:<12.3f} {avg:<12.2f} {min_time:<12.2f} {max_time:<12.2f}")
    
    def profile_code(self, func, *args, **kwargs):
        """Profile function with cProfile"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')
        stats.print_stats(20)
        
        return result

# Usage
profiler = PerformanceProfiler()

@profiler.measure("yolo_detection")
def detect_objects(image):
    return yolo_model.predict(image)

@profiler.measure("ocr_recognition")
def recognize_text(crop):
    return ocr_engine.recognize(crop)

# Run processing
for image_path in image_paths:
    image = cv2.imread(image_path)
    detections = detect_objects(image)
    for det in detections:
        text = recognize_text(det['crop'])

# Show report
profiler.report()
```

### 2. GPU Profiling

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_gpu_model(model, input_data):
    """Profile GPU operations"""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            output = model(input_data)
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export Chrome trace
    prof.export_chrome_trace("trace.json")
    
    return output

# Usage
input_tensor = torch.randn(16, 3, 640, 640).cuda()
output = profile_gpu_model(yolo_model, input_tensor)
```

### 3. End-to-End Benchmark

```python
import time
import numpy as np
from pathlib import Path

class EndToEndBenchmark:
    """Benchmark complete pipeline"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = {
            'latencies': [],
            'throughputs': [],
            'memory_peaks': []
        }
    
    def benchmark_single(self, image_path):
        """Benchmark single image"""
        import psutil
        import torch
        
        # Measure latency
        start = time.perf_counter()
        result = self.pipeline.process(image_path)
        end = time.perf_counter()
        
        latency = end - start
        self.results['latencies'].append(latency)
        
        # Memory
        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1e9
            self.results['memory_peaks'].append(memory)
            torch.cuda.reset_peak_memory_stats()
        
        return result
    
    def benchmark_batch(self, image_paths, batch_size=16):
        """Benchmark batch processing"""
        start = time.perf_counter()
        
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            batch_results = self.pipeline.process_batch(batch)
            results.extend(batch_results)
        
        end = time.perf_counter()
        
        total_time = end - start
        throughput = len(image_paths) / total_time
        self.results['throughputs'].append(throughput)
        
        return results
    
    def report(self):
        """Generate benchmark report"""
        latencies = np.array(self.results['latencies'])
        throughputs = np.array(self.results['throughputs'])
        
        print("\n=== Benchmark Report ===")
        print(f"\nLatency (single image):")
        print(f"  Mean:   {latencies.mean()*1000:.2f} ms")
        print(f"  Median: {np.median(latencies)*1000:.2f} ms")
        print(f"  P95:    {np.percentile(latencies, 95)*1000:.2f} ms")
        print(f"  P99:    {np.percentile(latencies, 99)*1000:.2f} ms")
        print(f"  Min:    {latencies.min()*1000:.2f} ms")
        print(f"  Max:    {latencies.max()*1000:.2f} ms")
        
        if throughputs.size > 0:
            print(f"\nThroughput (batch):")
            print(f"  Mean: {throughputs.mean():.2f} img/s")
            print(f"  Max:  {throughputs.max():.2f} img/s")
        
        if self.results['memory_peaks']:
            memory_peaks = np.array(self.results['memory_peaks'])
            print(f"\nGPU Memory:")
            print(f"  Mean Peak: {memory_peaks.mean():.2f} GB")
            print(f"  Max Peak:  {memory_peaks.max():.2f} GB")

# Usage
benchmark = EndToEndBenchmark(pipeline)

# Single image benchmarks
for image_path in test_images:
    benchmark.benchmark_single(image_path)

# Batch benchmark
benchmark.benchmark_batch(test_images, batch_size=16)

# Report
benchmark.report()
```

---

## Production Best Practices

### 1. Async API with FastAPI

```python
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn

app = FastAPI()

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# Model loading (singleton)
from src.pipeline import FullPipeline
pipeline = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global pipeline
    pipeline = FullPipeline()
    print("Models loaded")

@app.post("/process")
async def process_image(file: UploadFile, background_tasks: BackgroundTasks):
    """Process image asynchronously"""
    # Read image
    contents = await file.read()
    
    # Process in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        pipeline.process_bytes,
        contents
    )
    
    return JSONResponse(content=result)

@app.post("/process_batch")
async def process_batch(files: list[UploadFile]):
    """Process multiple images"""
    contents = await asyncio.gather(*[file.read() for file in files])
    
    # Process in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, pipeline.process_bytes, content)
        for content in contents
    ]
    results = await asyncio.gather(*tasks)
    
    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### 2. Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Database with connection pooling
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis with connection pooling
import redis
from redis.connection import ConnectionPool

redis_pool = ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50
)
redis_client = redis.Redis(connection_pool=redis_pool)
```

### 3. Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
requests_total = Counter('datalid_requests_total', 'Total requests')
request_duration = Histogram('datalid_request_duration_seconds', 'Request duration')
processing_errors = Counter('datalid_processing_errors_total', 'Processing errors')
gpu_memory = Gauge('datalid_gpu_memory_gb', 'GPU memory usage')
queue_size = Gauge('datalid_queue_size', 'Processing queue size')

class MonitoredPipeline:
    """Pipeline with Prometheus monitoring"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def process(self, image):
        """Process with monitoring"""
        requests_total.inc()
        
        start = time.time()
        try:
            result = self.pipeline.process(image)
            
            # Update metrics
            duration = time.time() - start
            request_duration.observe(duration)
            
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                gpu_memory.set(memory_gb)
            
            return result
            
        except Exception as e:
            processing_errors.inc()
            raise

# Start Prometheus metrics server
start_http_server(8001)

# Use monitored pipeline
monitored = MonitoredPipeline(pipeline)
```

### 4. Health Checks

```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import torch

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "checks": {}
    }
    
    # Check GPU
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            health_status["checks"]["gpu"] = "ok"
        except:
            health_status["checks"]["gpu"] = "error"
            health_status["status"] = "unhealthy"
    
    # Check models loaded
    try:
        if pipeline is not None:
            health_status["checks"]["models"] = "ok"
        else:
            health_status["checks"]["models"] = "not_loaded"
            health_status["status"] = "unhealthy"
    except:
        health_status["checks"]["models"] = "error"
        health_status["status"] = "unhealthy"
    
    # Check memory
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        health_status["checks"]["memory"] = {
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2)
        }
    
    status_code = status.HTTP_200_OK if health_status["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/readiness")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if pipeline is None:
        return JSONResponse(
            content={"status": "not_ready", "reason": "models_not_loaded"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    return JSONResponse(content={"status": "ready"})
```

---

## Configuration Examples

### High-Performance Configuration

```yaml
# config/optimization/high_performance.yaml
pipeline:
  name: high_performance
  
  # GPU settings
  gpu:
    enabled: true
    device_id: 0
    memory_fraction: 0.9
    mixed_precision: true
  
  # Batch processing
  batch:
    enabled: true
    max_size: 32
    max_wait_ms: 50
    dynamic_batching: true
  
  # Model settings
  yolo:
    model: yolov8n-seg.pt
    half_precision: true
    batch_size: 16
    
  ocr:
    engine: parseq
    batch_size: 32
    use_fp16: true
  
  # Caching
  cache:
    enabled: true
    type: redis
    ttl: 3600
    max_size_gb: 10
  
  # Preprocessing
  preprocessing:
    parallel_workers: 8
    gpu_accelerated: true
    
  # Monitoring
  monitoring:
    enabled: true
    prometheus_port: 8001
```

### Memory-Optimized Configuration

```yaml
# config/optimization/memory_optimized.yaml
pipeline:
  name: memory_optimized
  
  gpu:
    enabled: true
    memory_fraction: 0.7
    mixed_precision: true
  
  yolo:
    model: yolov8n-seg.pt
    half_precision: true
    batch_size: 8
    
  ocr:
    engine: parseq
    batch_size: 16
    
  cache:
    enabled: true
    type: disk
    max_size_gb: 5
  
  preprocessing:
    parallel_workers: 2
    lazy_loading: true
    
  cleanup:
    aggressive: true
    interval_seconds: 60
```

---

## Related Documentation

- [20-OCR-ENGINES.md](20-OCR-ENGINES.md) - OCR engine comparison and selection
- [19-YAML-CONFIG.md](19-YAML-CONFIG.md) - Complete configuration reference
- [24-BEST-PRACTICES.md](24-BEST-PRACTICES.md) - Best practices guide
- [15-DEPLOYMENT.md](15-DEPLOYMENT.md) - Production deployment
- [22-TROUBLESHOOTING.md](22-TROUBLESHOOTING.md) - Common issues

---

## Summary Checklist

### Quick Optimization Checklist

- [ ] Enable GPU if available
- [ ] Use mixed precision (FP16)
- [ ] Implement batch processing
- [ ] Configure optimal batch sizes
- [ ] Enable result caching
- [ ] Preload models
- [ ] Use parallel preprocessing
- [ ] Monitor GPU memory
- [ ] Profile bottlenecks
- [ ] Implement health checks

### Performance Targets

| Configuration | Throughput | Latency | Memory |
|--------------|-----------|---------|--------|
| **Development** | 5-10 img/s | < 1s | < 4GB |
| **Production** | 20-50 img/s | < 500ms | < 8GB |
| **High-Perf** | 100+ img/s | < 100ms | Optimized |

---

*For questions or support, see [22-TROUBLESHOOTING.md](22-TROUBLESHOOTING.md) or contact the team.*

---

*Last Updated: November 2025*
