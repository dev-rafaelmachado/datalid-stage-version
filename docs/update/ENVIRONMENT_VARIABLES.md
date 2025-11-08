# 🔧 Environment Variables Guide

## Overview

This document explains how environment variables are organized in the Datalid project to avoid conflicts between different components (Core, API, etc.).

## Variable Prefixes

The project uses **prefixes** to namespace environment variables:

| Prefix | Component | Config File | Description |
|--------|-----------|-------------|-------------|
| `API_` | API Server | `src/api/config.py` | All API-related settings |
| `CORE_` | Core System | `src/core/config.py` | Core system settings (optional) |

## Why Use Prefixes?

Without prefixes, environment variables from different components could conflict. For example:
- `HOST` could mean the API host OR a database host
- `DEBUG` could control API debug mode OR core system debug mode

By using prefixes like `API_HOST` and `CORE_HOST`, we eliminate ambiguity.

---

## API Configuration (`API_` prefix)

All API-related environment variables use the `API_` prefix.

### Server Settings

```bash
API_HOST=0.0.0.0              # Server host (default: 0.0.0.0)
API_PORT=8000                 # Server port (default: 8000)
API_WORKERS=1                 # Number of Uvicorn workers (default: 1)
API_RELOAD=false              # Auto-reload on code changes (dev only)
```

### API Information

```bash
API_NAME="Datalid API"
API_VERSION="1.0.0"
API_DESCRIPTION="API REST para detecção e extração de datas de validade"
```

### CORS Settings

```bash
API_CORS_ENABLED=true
API_CORS_ORIGINS=["*"]        # Allowed origins (use specific domains in production)
API_CORS_CREDENTIALS=true
API_CORS_METHODS=["*"]
API_CORS_HEADERS=["*"]
```

### Security & Limits

```bash
API_MAX_FILE_SIZE_MB=10.0     # Max upload file size in MB
API_MAX_BATCH_SIZE=50         # Max number of files in batch request
API_ALLOWED_EXTENSIONS=[".jpg", ".jpeg", ".png", ".bmp"]
```

### Rate Limiting

```bash
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS=60    # Requests per minute per IP
```

### Authentication

```bash
API_AUTH_ENABLED=false        # Enable API key authentication
API_API_KEY_HEADER=X-API-Key  # Header name for API key
API_API_KEYS=[]               # List of valid API keys (JSON array)
```

Example with multiple API keys:
```bash
API_API_KEYS=["key1_abc123", "key2_def456"]
```

### Model Configuration

```bash
API_DEFAULT_YOLO_MODEL=experiments/yolov8m_seg_best/weights/best.pt
API_MODEL_DEVICE=0            # CUDA device (0, 1, 2...) or "cpu"
```

### OCR Configuration

```bash
API_DEFAULT_OCR_ENGINE=openocr
API_DEFAULT_OCR_CONFIG=config/ocr/openocr.yaml
API_DEFAULT_PREPROCESSING_CONFIG=config/preprocessing/ppro-openocr-medium.yaml
```

### Detection Parameters

```bash
API_DEFAULT_CONFIDENCE=0.25   # Confidence threshold (0.0-1.0)
API_DEFAULT_IOU=0.7           # IoU threshold for NMS
```

### Cache Settings

```bash
API_CACHE_ENABLED=true
API_CACHE_TTL=3600            # Cache time-to-live in seconds
API_CACHE_MAX_SIZE=100        # Max number of cached items
```

### Storage Settings

```bash
API_UPLOAD_DIR=uploads
API_RESULTS_DIR=outputs/api_results
API_CLEANUP_UPLOADS=true      # Delete uploads after processing
API_SAVE_RESULTS_BY_DEFAULT=false
```

### Logging

```bash
API_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR, CRITICAL
API_LOG_REQUESTS=true         # Log all HTTP requests
API_LOG_FILE=logs/api.log
```

### Monitoring

```bash
API_ENABLE_METRICS=true
API_METRICS_ENDPOINT=/metrics # Prometheus metrics endpoint
```

### Documentation Endpoints

```bash
API_DOCS_URL=/docs            # Swagger UI
API_REDOC_URL=/redoc          # ReDoc UI
API_OPENAPI_URL=/openapi.json # OpenAPI schema
```

### Development Settings

```bash
API_DEBUG=false               # Enable debug mode
API_ENVIRONMENT=production    # Environment: development, staging, production
```

---

## Core Configuration (`CORE_` prefix)

Core system settings (if needed) use the `CORE_` prefix. Most core settings are defined in code and don't need environment variables.

Example:
```bash
CORE_TRAIN_SPLIT=0.7
CORE_VAL_SPLIT=0.2
CORE_TEST_SPLIT=0.1
```

---

## Setup Instructions

### 1. Copy the Example File

```bash
cp .env.example .env
```

### 2. Edit Your `.env` File

Customize the values according to your environment:

```bash
# Development
API_HOST=127.0.0.1
API_PORT=8000
API_DEBUG=true
API_RELOAD=true
API_ENVIRONMENT=development

# Enable auth for production
API_AUTH_ENABLED=true
API_API_KEYS=["your-secret-key-here"]
```

### 3. Secure Your `.env` File

**Important:** Never commit `.env` files to version control!

The `.env` file is already in `.gitignore`, but double-check:

```bash
# Make sure .env is ignored
cat .gitignore | grep ".env"
```

---

## Production Deployment

### Security Checklist

When deploying to production, ensure:

✅ **Change default secrets:**
```bash
API_API_KEYS=["your-strong-random-key-here"]
```

✅ **Restrict CORS origins:**
```bash
API_CORS_ORIGINS=["https://yourdomain.com"]
```

✅ **Enable authentication:**
```bash
API_AUTH_ENABLED=true
```

✅ **Set proper log level:**
```bash
API_LOG_LEVEL=WARNING
API_DEBUG=false
```

✅ **Configure proper limits:**
```bash
API_RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_REQUESTS=60
API_MAX_FILE_SIZE_MB=10.0
```

### Docker Environment

When using Docker, pass environment variables:

```bash
# Using docker run
docker run -e API_PORT=8000 -e API_HOST=0.0.0.0 datalid-api

# Using docker-compose (in docker-compose.yml)
services:
  api:
    environment:
      - API_PORT=8000
      - API_HOST=0.0.0.0
      - API_AUTH_ENABLED=true
      - API_API_KEYS=["${API_KEY}"]
```

Or use an env file:
```yaml
services:
  api:
    env_file:
      - .env.production
```

---

## Troubleshooting

### Pydantic Validation Error

If you see errors like:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Config
HOST
  Extra inputs are not permitted
```

**Cause:** The variable name doesn't have the correct prefix.

**Solution:** Make sure all API variables use the `API_` prefix:
```bash
# ❌ Wrong
HOST=0.0.0.0

# ✅ Correct
API_HOST=0.0.0.0
```

### Variable Not Being Read

**Cause:** Wrong prefix or typo in variable name.

**Solution:** 
1. Check the variable name matches the config field name (case-insensitive)
2. Ensure you're using the correct prefix
3. Restart the API server after changing `.env`

### Config Field Names vs. Environment Variables

The config class uses `snake_case` field names, and Pydantic automatically matches them to environment variables:

| Config Field | Environment Variable |
|--------------|---------------------|
| `host` | `API_HOST` |
| `port` | `API_PORT` |
| `api_name` | `API_API_NAME` |
| `cors_enabled` | `API_CORS_ENABLED` |

The `API_` prefix is automatically stripped by Pydantic's `env_prefix` setting.

---

## Testing Your Configuration

### View Current Config

Start the API and check the `/health` endpoint:

```bash
curl http://localhost:8000/health
```

This shows the current configuration (sensitive values are masked).

### Verify Environment Variables

In Python:
```python
from src.api.config import settings

print(f"API running on {settings.host}:{settings.port}")
print(f"Auth enabled: {settings.auth_enabled}")
print(f"Debug mode: {settings.debug}")
```

---

## Best Practices

1. **Use `.env` for local development**
   - Keep your local config in `.env` (not committed)
   - Use `.env.example` as a template

2. **Use separate env files for different environments**
   ```
   .env.development
   .env.staging
   .env.production
   ```

3. **Use environment variables in CI/CD**
   - Set secrets in GitHub Actions, GitLab CI, etc.
   - Never hardcode secrets in code

4. **Document new variables**
   - Update `.env.example` when adding new config
   - Update this documentation

5. **Validate on startup**
   - Pydantic automatically validates all settings
   - The API won't start if config is invalid

---

## Reference

- **Pydantic Settings:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **FastAPI Config:** https://fastapi.tiangolo.com/advanced/settings/
- **Twelve-Factor App:** https://12factor.net/config

---

**Last Updated:** 2024
**Maintainer:** Datalid Team
