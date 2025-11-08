# 🏗️ Arquitetura da Datalid API

## 📊 Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENTES                                 │
├─────────────────────────────────────────────────────────────────┤
│  Browser  │  Python SDK  │  cURL/HTTP  │  WebSocket  │  Mobile │
└─────┬───────────┬──────────────┬────────────┬───────────────┬───┘
      │           │              │            │               │
      └───────────┴──────────────┴────────────┴───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   LOAD BALANCER    │
                    │   (opcional)       │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ Worker 1│    │ Worker 2│    │ Worker N│
         └────┬────┘    └────┬────┘    └────┬────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    FASTAPI APP     │
                    │   (main.py)        │
                    └─────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
      ┌────▼────┐      ┌─────▼──────┐    ┌─────▼──────┐
      │ Routes  │      │ Middleware │    │  WebSocket │
      │  v1/v2  │      │  (CORS,    │    │  Handler   │
      │         │      │  Auth, etc)│    │            │
      └────┬────┘      └─────┬──────┘    └─────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   SERVICE LAYER    │
                    │  (service.py)      │
                    └─────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
      ┌────▼────┐      ┌─────▼──────┐    ┌─────▼──────┐
      │  Jobs   │      │  Metrics   │    │    Auth    │
      │ Manager │      │ Collector  │    │  Manager   │
      └────┬────┘      └─────┬──────┘    └─────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FULL PIPELINE     │
                    │  (Existing)        │
                    └─────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
      ┌────▼────┐      ┌─────▼──────┐    ┌─────▼──────┐
      │  YOLO   │      │    OCR     │    │   Parser   │
      │Detection│      │  Engines   │    │  (Dates)   │
      └────┬────┘      └─────┬──────┘    └─────┬──────┘
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │     RESULTADO      │
                    └────────────────────┘
```

## 🔄 Fluxo de Processamento

### 1. Processamento Síncrono (POST /process)

```
Cliente
  │
  ├─► POST /process + imagem
  │
API
  │
  ├─► Middleware (Auth, CORS, Timing)
  │
  ├─► Validação (formato, tamanho)
  │
  ├─► Service.process_image()
  │     │
  │     ├─► FullPipeline.process()
  │     │     │
  │     │     ├─► YOLO Detection
  │     │     ├─► OCR Engine
  │     │     └─► Date Parser
  │     │
  │     ├─► Convert Results
  │     ├─► Generate Visualization (opcional)
  │     └─► Extract Crops (opcional)
  │
  ├─► Record Metrics
  │
  └─► Return Response
```

### 2. Processamento Assíncrono (POST /v2/jobs)

```
Cliente
  │
  ├─► POST /v2/jobs + imagem
  │
API
  │
  ├─► Create Job (job_id)
  │
  ├─► Start Background Task
  │     │
  │     └─► Process (mesmo fluxo síncrono)
  │
  └─► Return job_id imediatamente
  
Cliente (polling)
  │
  ├─► GET /v2/jobs/{job_id}
  │
  └─► Recebe status + resultado quando completo
```

### 3. Processamento via WebSocket (WS /v2/ws)

```
Cliente
  │
  ├─► Connect WS /v2/ws
  │
  ├─► Send: {type: "process", image: "base64..."}
  │
API
  │
  ├─► Accept Connection
  │
  ├─► Process com Feedback
  │     │
  │     ├─► Send: {type: "progress", step: "detection", progress: 0.3}
  │     ├─► Send: {type: "progress", step: "ocr", progress: 0.6}
  │     ├─► Send: {type: "progress", step: "parsing", progress: 0.9}
  │     └─► Send: {type: "result", data: {...}}
  │
  └─► Close Connection
```

## 📦 Componentes Detalhados

### Main Application (main.py)
```
FastAPI App
├── Exception Handlers
├── Middleware Setup
├── Route Registration
├── Startup Events
│   ├── Initialize Service
│   ├── Create Directories
│   └── Load Models
└── Shutdown Events
    └── Cleanup
```

### Routes (routes.py + routes_v2.py)
```
API v1                      API v2
├── GET /                   ├── WS /v2/ws
├── GET /health             ├── POST /v2/jobs
├── GET /info               ├── GET /v2/jobs/{id}
├── POST /process           ├── GET /v2/jobs
├── POST /process/batch     ├── DELETE /v2/jobs/{id}
└── POST /process/url       ├── GET /v2/metrics
                            ├── GET /v2/metrics/endpoints
                            ├── GET /v2/metrics/prometheus
                            └── POST /v2/admin/*
```

### Service Layer (service.py)
```
ProcessingService
├── pipeline: FullPipeline
├── initialize()
├── process_image()
├── _convert_pipeline_result()
├── _convert_detection()
├── _convert_ocr_result()
├── _convert_date()
├── _create_visualization()
├── _extract_crops()
└── _save_results()
```

### Job System (jobs.py)
```
JobManager
├── jobs: Dict[job_id, Job]
├── create_job()
├── get_job()
├── update_job()
├── cancel_job()
├── delete_job()
├── list_jobs()
├── process_job_async()
└── _cleanup_old_jobs()
```

### Metrics (metrics.py)
```
MetricsCollector
├── Counters
│   ├── total_requests
│   ├── total_errors
│   ├── total_images_processed
│   └── total_dates_found
├── Gauges
│   └── active_connections
├── Histograms
│   ├── processing_times
│   └── request_times
├── Methods
│   ├── record_request()
│   ├── record_processing()
│   ├── get_metrics()
│   └── get_prometheus_metrics()
```

### Authentication (auth.py)
```
Auth System
├── API Key
│   ├── verify_api_key()
│   └── Header: X-API-Key
├── JWT
│   ├── create_access_token()
│   ├── verify_token()
│   └── get_current_user()
└── Scopes/Permissions
    └── require_scope()
```

## 🔌 Integrações

### Existing Pipeline
```
API Service
     │
     └─► FullPipeline (src/pipeline/full_pipeline.py)
            │
            ├─► Detection (YOLO)
            ├─► OCR (múltiplos engines)
            ├─► Preprocessing
            ├─► Postprocessing
            └─► Date Parsing
```

### External Services (opcional)
```
API
 │
 ├─► Prometheus (métricas)
 ├─► Redis (cache - futuro)
 ├─► Database (persistência - futuro)
 └─► S3/Storage (imagens - futuro)
```

## 📊 Data Flow

### Request Data Flow
```
Client Request
     │
     ├─► Pydantic Validation
     │     (schemas.py)
     │
     ├─► Business Logic
     │     (service.py)
     │
     ├─► ML Processing
     │     (pipeline)
     │
     └─► Response Serialization
           (schemas.py)
```

### Configuration Flow
```
Environment Variables (.env)
     │
     ├─► Pydantic Settings (config.py)
     │
     ├─► Dependency Injection
     │
     └─► Service Configuration
```

## 🔒 Security Layers

```
Request
  │
  ├─► CORS Middleware
  │     (verifica origin)
  │
  ├─► Rate Limiting
  │     (limita requisições)
  │
  ├─► Authentication
  │     (API Key ou JWT)
  │
  ├─► Input Validation
  │     (Pydantic schemas)
  │
  ├─► File Validation
  │     (tipo, tamanho, conteúdo)
  │
  └─► Process
```

## 📈 Scaling Strategy

### Horizontal Scaling
```
Load Balancer
     │
     ├─► API Instance 1
     ├─► API Instance 2
     ├─► API Instance 3
     └─► API Instance N
            │
            └─► Shared Storage
                  │
                  ├─► Models (read-only)
                  └─► Results (write)
```

### Vertical Scaling
```
Single Instance
├── Multiple Workers (uvicorn)
├── GPU Acceleration
├── Batch Processing
└── Async Jobs
```

## 🎯 Performance Optimization Points

1. **Model Loading** - Cache em memória (singleton)
2. **Image Processing** - GPU acceleration
3. **Batch Processing** - Processar múltiplas imagens
4. **Async Jobs** - Não bloquear requisições
5. **Response Caching** - Cache de resultados (futuro)
6. **Connection Pooling** - Reuso de conexões

## 🔍 Monitoring Points

```
Metrics Collection
├── Request Level
│   ├── Duration
│   ├── Status Code
│   └── Endpoint
├── Processing Level
│   ├── Detection Time
│   ├── OCR Time
│   └── Total Time
└── System Level
    ├── Memory Usage
    ├── CPU Usage
    └── GPU Usage
```

## 🚀 Deployment Options

### Development
```bash
python scripts/api/start_server.py --dev
```

### Production - Single Server
```bash
python scripts/api/start_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Production - Docker
```bash
docker run -p 8000:8000 \
  --gpus all \
  datalid-api
```

### Production - Kubernetes
```yaml
Deployment
├── Pods (replicas)
├── Service (load balancer)
├── Ingress (routing)
└── ConfigMap (settings)
```

---

Esta arquitetura segue as **melhores práticas de mercado** para APIs modernas:
- ✅ Modular e desacoplada
- ✅ Fácil de testar
- ✅ Fácil de escalar
- ✅ Observável
- ✅ Segura
- ✅ Bem documentada
