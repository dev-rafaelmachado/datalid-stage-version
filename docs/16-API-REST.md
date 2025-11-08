# 16. REST API Documentation

## 📋 Overview

The Datalid 3.0 REST API provides a simple and powerful interface for date detection and OCR processing in documents. This API is built with FastAPI and follows REST principles for ease of integration.

---

## 🎯 Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Formats](#requestresponse-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Batch Processing](#batch-processing)
8. [WebSocket Support](#websocket-support)
9. [API Client Examples](#api-client-examples)
10. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Starting the API Server

```bash
# Using Makefile
make api-start

# Or directly
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# With custom configuration
API_HOST=0.0.0.0 API_PORT=8080 python -m uvicorn src.api.main:app
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Process a single image
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@image.jpg" \
  -F "ocr_engine=paddleocr"
```

### Interactive Documentation

Once the API is running, access the interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🔐 Authentication

### API Key Authentication (Optional)

If enabled, requests must include an API key:

```bash
# In headers
curl -H "X-API-Key: your_api_key_here" \
  http://localhost:8000/api/v1/process
```

### Configuration

```yaml
# .env or environment variables
API_KEY_ENABLED=true
API_KEY=your_secret_api_key_here
API_KEY_HEADER=X-API-Key
```

### Token-Based Authentication (Advanced)

For production deployments, JWT tokens can be used:

```bash
# Get token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/process
```

---

## 🛣️ API Endpoints

### Health & Info

#### `GET /`
Root endpoint with API information.

**Response:**
```json
{
  "name": "Datalid API",
  "version": "3.0.0",
  "description": "API REST para detecção de datas de validade",
  "status": "online",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": 3600,
  "version": "3.0.0",
  "components": {
    "processing_service": {
      "status": "healthy",
      "message": "Pronto"
    },
    "pipeline": {
      "status": "healthy",
      "message": "Carregado"
    }
  }
}
```

#### `GET /metrics`
Prometheus-compatible metrics endpoint.

**Response:**
```
# HELP datalid_requests_total Total number of requests
# TYPE datalid_requests_total counter
datalid_requests_total{method="POST",endpoint="/api/v1/process"} 42

# HELP datalid_processing_duration_seconds Processing duration
# TYPE datalid_processing_duration_seconds histogram
datalid_processing_duration_seconds_bucket{le="0.5"} 30
```

---

### Image Processing

#### `POST /api/v1/process`
Process a single image for date detection.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Image file (JPEG, PNG, BMP, TIFF) |
| ocr_engine | String | No | OCR engine: `tesseract`, `paddleocr`, `easyocr`, `parseq`, `trocr`, `openocr` (default: `openocr`) |
| confidence | Float | No | Detection confidence threshold (0.0-1.0, default: 0.25) |
| preprocessing | Boolean | No | Enable preprocessing (default: true) |
| return_debug_images | Boolean | No | Return annotated images (default: false) |
| date_formats | List[String] | No | Expected date formats (default: auto-detect) |

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@invoice.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "confidence=0.3" \
  -F "preprocessing=true" \
  -F "return_debug_images=true"
```

**Example Response:**

```json
{
  "status": "success",
  "message": "Processamento concluído com sucesso",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123abc",
  "processing_time": 0.234,
  "data": {
    "image_info": {
      "width": 1920,
      "height": 1080,
      "format": "JPEG",
      "size_bytes": 245678
    },
    "detections": [
      {
        "id": "det_001",
        "class": "expiry_date",
        "confidence": 0.87,
        "bbox": {
          "x1": 100,
          "y1": 200,
          "x2": 300,
          "y2": 250
        },
        "text": "Validade: 15/12/2025",
        "dates": [
          {
            "value": "2025-12-15",
            "format": "DD/MM/YYYY",
            "confidence": 0.92,
            "normalized": "2025-12-15T00:00:00Z",
            "relative": "em 1 ano e 11 meses"
          }
        ]
      }
    ],
    "statistics": {
      "total_detections": 1,
      "valid_dates": 1,
      "average_confidence": 0.87,
      "ocr_engine": "paddleocr"
    }
  },
  "debug_images": {
    "annotated": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
  }
}
```

---

#### `POST /api/v1/process/url`
Process an image from a URL.

**Request Body:**
```json
{
  "url": "https://example.com/image.jpg",
  "ocr_engine": "paddleocr",
  "confidence": 0.25,
  "preprocessing": true
}
```

**Response:** Same as `/api/v1/process`

---

#### `POST /api/v1/process/batch`
Process multiple images in batch.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| files | List[File] | Yes | Multiple image files |
| ocr_engine | String | No | OCR engine for all images |
| confidence | Float | No | Detection confidence threshold |
| parallel | Boolean | No | Process in parallel (default: true) |
| max_workers | Integer | No | Number of parallel workers (default: 4) |

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/process/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "parallel=true" \
  -F "max_workers=4"
```

**Example Response:**

```json
{
  "status": "success",
  "message": "Processamento em lote concluído",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_batch_456",
  "total_processing_time": 1.23,
  "data": {
    "total_images": 3,
    "successful": 3,
    "failed": 0,
    "results": [
      {
        "filename": "image1.jpg",
        "status": "success",
        "processing_time": 0.45,
        "detections": [...]
      },
      {
        "filename": "image2.jpg",
        "status": "success",
        "processing_time": 0.38,
        "detections": [...]
      },
      {
        "filename": "image3.jpg",
        "status": "success",
        "processing_time": 0.40,
        "detections": [...]
      }
    ],
    "statistics": {
      "total_detections": 5,
      "average_confidence": 0.88,
      "average_processing_time": 0.41
    }
  }
}
```

---

### Asynchronous Processing

#### `POST /api/v1/jobs`
Submit a job for asynchronous processing.

**Request Body:**
```json
{
  "images": [
    {"url": "https://example.com/image1.jpg"},
    {"url": "https://example.com/image2.jpg"}
  ],
  "config": {
    "ocr_engine": "paddleocr",
    "confidence": 0.25
  },
  "callback_url": "https://your-app.com/webhook"
}
```

**Response:**
```json
{
  "job_id": "job_789xyz",
  "status": "queued",
  "created_at": "2024-01-15T10:30:00Z",
  "status_url": "/api/v1/jobs/job_789xyz"
}
```

#### `GET /api/v1/jobs/{job_id}`
Get job status and results.

**Response:**
```json
{
  "job_id": "job_789xyz",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "progress": {
    "total": 2,
    "completed": 2,
    "failed": 0,
    "percentage": 100
  },
  "results": {
    "images": [...]
  }
}
```

---

### Configuration

#### `GET /api/v1/config`
Get current API configuration.

**Response:**
```json
{
  "available_ocr_engines": [
    "tesseract",
    "paddleocr",
    "easyocr",
    "parseq",
    "trocr",
    "openocr"
  ],
  "default_ocr_engine": "openocr",
  "max_file_size": 10485760,
  "max_batch_size": 10,
  "supported_formats": ["jpeg", "jpg", "png", "bmp", "tiff"],
  "rate_limit": {
    "requests_per_minute": 60,
    "requests_per_hour": 1000
  }
}
```

#### `POST /api/v1/config/reload`
Reload configuration (requires admin privileges).

**Response:**
```json
{
  "status": "success",
  "message": "Configuração recarregada com sucesso"
}
```

---

## 📦 Request/Response Formats

### Standard Response Format

All API responses follow this structure:

```json
{
  "status": "success|partial|failed",
  "message": "Human-readable message",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "unique_request_id",
  "processing_time": 0.234,
  "data": {
    // Response data here
  },
  "metadata": {
    // Optional metadata
  }
}
```

### Error Response Format

```json
{
  "status": "error",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_error_123",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {
      // Additional error context
    }
  }
}
```

### Detection Object

```json
{
  "id": "det_001",
  "class": "expiry_date",
  "confidence": 0.87,
  "bbox": {
    "x1": 100,
    "y1": 200,
    "x2": 300,
    "y2": 250,
    "width": 200,
    "height": 50
  },
  "segmentation": [[x1, y1, x2, y2, ...]],
  "text": "Validade: 15/12/2025",
  "text_confidence": 0.92,
  "dates": [
    {
      "value": "2025-12-15",
      "format": "DD/MM/YYYY",
      "confidence": 0.92,
      "normalized": "2025-12-15T00:00:00Z",
      "relative": "em 1 ano e 11 meses",
      "is_expired": false,
      "days_until_expiry": 700
    }
  ]
}
```

---

## ⚠️ Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created (for job submissions) |
| 202 | Accepted (async processing) |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (missing/invalid API key) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found |
| 413 | Payload Too Large |
| 422 | Unprocessable Entity (validation error) |
| 429 | Too Many Requests (rate limit exceeded) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `VALIDATION_ERROR` | Invalid request parameters | Check request format |
| `INVALID_IMAGE` | Invalid or corrupted image | Verify image file |
| `FILE_TOO_LARGE` | File exceeds size limit | Reduce file size |
| `UNSUPPORTED_FORMAT` | Unsupported image format | Use JPEG, PNG, BMP, or TIFF |
| `OCR_ENGINE_ERROR` | OCR processing failed | Try different OCR engine |
| `DETECTION_ERROR` | YOLO detection failed | Check image quality |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Slow down requests |
| `INTERNAL_ERROR` | Server error | Contact support |

### Error Examples

#### Validation Error
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Erro de validação dos dados de entrada",
    "details": {
      "errors": [
        {
          "loc": ["body", "confidence"],
          "msg": "ensure this value is less than or equal to 1.0",
          "type": "value_error"
        }
      ]
    }
  }
}
```

#### File Too Large
```json
{
  "status": "error",
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "Arquivo excede o tamanho máximo permitido",
    "details": {
      "file_size": 15728640,
      "max_size": 10485760,
      "max_size_mb": 10
    }
  }
}
```

---

## 🚦 Rate Limiting

### Default Limits

- **Per IP**: 60 requests/minute
- **Per API Key**: 1000 requests/hour
- **Concurrent**: 10 simultaneous requests

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1610712000
Retry-After: 15
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url, **kwargs):
    """Make request with automatic retry on rate limit."""
    max_retries = 3
    
    for attempt in range(max_retries):
        response = requests.post(url, **kwargs)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

---

## 📦 Batch Processing

### Best Practices

1. **Batch Size**: Keep batches under 10 images for optimal performance
2. **Parallel Processing**: Enable for faster throughput
3. **Error Handling**: Handle partial failures gracefully
4. **Memory**: Monitor memory usage for large batches

### Example: Processing Multiple Files

```python
import requests

def process_batch(image_paths, ocr_engine="paddleocr"):
    """Process multiple images in a batch."""
    url = "http://localhost:8000/api/v1/process/batch"
    
    files = [
        ("files", (path.name, open(path, "rb"), "image/jpeg"))
        for path in image_paths
    ]
    
    data = {
        "ocr_engine": ocr_engine,
        "parallel": "true",
        "max_workers": "4"
    }
    
    response = requests.post(url, files=files, data=data)
    
    # Close file handles
    for _, (_, f, _) in files:
        f.close()
    
    return response.json()

# Usage
from pathlib import Path

images = list(Path("images").glob("*.jpg"))
results = process_batch(images)

print(f"Processed: {results['data']['successful']}/{results['data']['total_images']}")
```

---

## 🔌 WebSocket Support

### Real-Time Processing Updates

Connect to WebSocket for real-time processing updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/process');

ws.onopen = () => {
    console.log('Connected to WebSocket');
    
    // Send image for processing
    ws.send(JSON.stringify({
        type: 'process',
        data: {
            image_base64: base64Image,
            ocr_engine: 'paddleocr'
        }
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'progress':
            console.log(`Progress: ${message.data.percentage}%`);
            break;
        case 'result':
            console.log('Processing complete:', message.data);
            break;
        case 'error':
            console.error('Error:', message.data);
            break;
    }
};
```

### Python WebSocket Client

```python
import asyncio
import websockets
import json
import base64

async def process_with_websocket(image_path):
    """Process image via WebSocket."""
    uri = "ws://localhost:8000/ws/process"
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    async with websockets.connect(uri) as websocket:
        # Send processing request
        await websocket.send(json.dumps({
            "type": "process",
            "data": {
                "image_base64": image_data,
                "ocr_engine": "paddleocr"
            }
        }))
        
        # Receive updates
        while True:
            message = json.loads(await websocket.recv())
            
            if message["type"] == "progress":
                print(f"Progress: {message['data']['percentage']}%")
            elif message["type"] == "result":
                print("Complete!")
                return message["data"]
            elif message["type"] == "error":
                raise Exception(message["data"]["message"])

# Usage
result = asyncio.run(process_with_websocket("image.jpg"))
```

---

## 💻 API Client Examples

### Python Requests

```python
import requests
from pathlib import Path

class DatalidClient:
    """Simple Datalid API client."""
    
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def process_image(self, image_path, **options):
        """Process a single image."""
        url = f"{self.base_url}/api/v1/process"
        
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                url,
                files=files,
                data=options,
                headers=self.headers
            )
        
        response.raise_for_status()
        return response.json()
    
    def process_batch(self, image_paths, **options):
        """Process multiple images."""
        url = f"{self.base_url}/api/v1/process/batch"
        
        files = [
            ("files", (Path(p).name, open(p, "rb"), "image/jpeg"))
            for p in image_paths
        ]
        
        response = requests.post(
            url,
            files=files,
            data=options,
            headers=self.headers
        )
        
        # Close files
        for _, (_, f, _) in files:
            f.close()
        
        response.raise_for_status()
        return response.json()

# Usage
client = DatalidClient(api_key="your_api_key")

# Process single image
result = client.process_image(
    "invoice.jpg",
    ocr_engine="paddleocr",
    confidence=0.3
)

print(f"Found {len(result['data']['detections'])} dates")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class DatalidClient {
    constructor(baseURL = 'http://localhost:8000', apiKey = null) {
        this.client = axios.create({
            baseURL: baseURL,
            headers: apiKey ? { 'X-API-Key': apiKey } : {}
        });
    }
    
    async processImage(imagePath, options = {}) {
        const form = new FormData();
        form.append('file', fs.createReadStream(imagePath));
        
        Object.entries(options).forEach(([key, value]) => {
            form.append(key, value);
        });
        
        const response = await this.client.post(
            '/api/v1/process',
            form,
            { headers: form.getHeaders() }
        );
        
        return response.data;
    }
    
    async processBatch(imagePaths, options = {}) {
        const form = new FormData();
        
        imagePaths.forEach(path => {
            form.append('files', fs.createReadStream(path));
        });
        
        Object.entries(options).forEach(([key, value]) => {
            form.append(key, value);
        });
        
        const response = await this.client.post(
            '/api/v1/process/batch',
            form,
            { headers: form.getHeaders() }
        );
        
        return response.data;
    }
}

// Usage
const client = new DatalidClient('http://localhost:8000', 'your_api_key');

client.processImage('invoice.jpg', {
    ocr_engine: 'paddleocr',
    confidence: 0.3
}).then(result => {
    console.log(`Found ${result.data.detections.length} dates`);
}).catch(error => {
    console.error('Error:', error.message);
});
```

### cURL Examples

```bash
# Simple processing
curl -X POST http://localhost:8000/api/v1/process \
  -H "X-API-Key: your_api_key" \
  -F "file=@image.jpg" \
  -F "ocr_engine=paddleocr"

# With all options
curl -X POST http://localhost:8000/api/v1/process \
  -H "X-API-Key: your_api_key" \
  -F "file=@image.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "confidence=0.3" \
  -F "preprocessing=true" \
  -F "return_debug_images=true" \
  -F "date_formats=DD/MM/YYYY,YYYY-MM-DD"

# Batch processing
curl -X POST http://localhost:8000/api/v1/process/batch \
  -H "X-API-Key: your_api_key" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "ocr_engine=paddleocr" \
  -F "parallel=true"

# Process from URL
curl -X POST http://localhost:8000/api/v1/process/url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "url": "https://example.com/image.jpg",
    "ocr_engine": "paddleocr",
    "confidence": 0.25
  }'
```

---

## ✅ Best Practices

### 1. Image Optimization

```python
from PIL import Image

def optimize_image(image_path, max_size=2000):
    """Optimize image before sending to API."""
    img = Image.open(image_path)
    
    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save optimized
    output_path = "optimized_" + Path(image_path).name
    img.save(output_path, 'JPEG', quality=85, optimize=True)
    
    return output_path
```

### 2. Error Handling

```python
def robust_api_call(client, image_path, max_retries=3):
    """Make API call with robust error handling."""
    from requests.exceptions import RequestException
    import time
    
    for attempt in range(max_retries):
        try:
            result = client.process_image(image_path)
            return result
            
        except requests.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
            elif e.response.status_code >= 500:  # Server error
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise  # Client error, don't retry
                
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    raise Exception("Max retries exceeded")
```

### 3. Async Processing for Large Batches

```python
import asyncio
import aiohttp

async def process_image_async(session, url, image_path):
    """Process image asynchronously."""
    with open(image_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename=image_path.name)
        data.add_field('ocr_engine', 'paddleocr')
        
        async with session.post(url, data=data) as response:
            return await response.json()

async def process_many_images(image_paths):
    """Process many images concurrently."""
    url = "http://localhost:8000/api/v1/process"
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_image_async(session, url, path)
            for path in image_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# Usage
results = asyncio.run(process_many_images(image_paths))
```

### 4. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    """API configuration."""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    ocr_engine: str = "paddleocr"
    confidence: float = 0.25
    preprocessing: bool = True
    timeout: int = 30
    max_retries: int = 3

# Load from environment
import os

config = APIConfig(
    base_url=os.getenv("DATALID_API_URL", "http://localhost:8000"),
    api_key=os.getenv("DATALID_API_KEY"),
    ocr_engine=os.getenv("DATALID_OCR_ENGINE", "paddleocr")
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status

# Check port binding
netstat -tulpn | grep 8000
```

#### Slow Processing
```python
# Enable debug mode to see timing
response = client.process_image(
    "image.jpg",
    return_timing=True
)

print(response['metadata']['timing'])
# {'yolo': 0.05, 'ocr': 0.15, 'preprocessing': 0.02}
```

#### Out of Memory
```yaml
# Reduce batch size
batch_size: 4  # Instead of 16

# Use smaller model
yolo:
  model: yolov8n-seg  # Instead of yolov8m

# Disable preprocessing for some images
preprocessing: false
```

---

## 📚 Related Documentation

- [01. Quick Start](01-QUICK-START.md) - Getting started
- [17. Python Client](17-PYTHON-CLIENT.md) - Full Python SDK
- [18. Integrations](18-INTEGRATIONS.md) - Integration examples
- [22. Troubleshooting](22-TROUBLESHOOTING.md) - Common issues

---

**Next Steps:**
- [Explore the Python SDK](17-PYTHON-CLIENT.md)
- [See integration examples](18-INTEGRATIONS.md)
- [Learn about best practices](24-BEST-PRACTICES.md)
