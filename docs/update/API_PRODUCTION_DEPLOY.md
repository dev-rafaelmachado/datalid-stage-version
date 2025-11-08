# 🚀 Deploy em Produção - Datalid API

Guia completo para fazer deploy da API em produção.

## 📋 Checklist Pré-Deploy

- [ ] Testes passando
- [ ] Modelos treinados disponíveis
- [ ] Configurações de produção definidas
- [ ] Secrets configurados
- [ ] Monitoramento configurado
- [ ] Backup strategy definida
- [ ] Logs configurados
- [ ] SSL/TLS configurado (se público)

## ⚙️ Configuração de Produção

### 1. Arquivo .env

```env
# IMPORTANTE: Use valores seguros!
ENVIRONMENT=production
DEBUG=false

# Servidor
HOST=0.0.0.0
PORT=8000
WORKERS=4  # ou número de CPUs

# Segurança
AUTH_ENABLED=true
API_KEYS=["GENERATE-SECURE-KEY-1","GENERATE-SECURE-KEY-2"]
JWT_SECRET_KEY=USE-openssl-rand-hex-32

# Performance
MODEL_DEVICE=0  # GPU
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=50
RATE_LIMIT_REQUESTS=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/datalid/api.log

# Storage (use paths absolutos)
UPLOAD_DIR=/tmp/datalid/uploads
RESULTS_DIR=/var/datalid/results
```

### 2. Gerar Secrets Seguros

```bash
# API Keys (use valores únicos)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# JWT Secret
openssl rand -hex 32
```

## 🐳 Deploy com Docker

### Dockerfile (já existe, mas otimizado)

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/uploads /app/results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Build e Run

```bash
# Build
docker build -t datalid-api:latest .

# Run (CPU)
docker run -d \
  --name datalid-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  -v datalid-uploads:/app/uploads \
  -v datalid-results:/app/results \
  -v datalid-logs:/app/logs \
  --env-file .env.production \
  --restart unless-stopped \
  datalid-api:latest

# Run (GPU)
docker run -d \
  --name datalid-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  -v datalid-uploads:/app/uploads \
  -v datalid-results:/app/results \
  -v datalid-logs:/app/logs \
  --env-file .env.production \
  --restart unless-stopped \
  datalid-api:latest
```

## 🐙 Deploy com Docker Compose

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: datalid-api:latest
    container_name: datalid-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
      - datalid-uploads:/app/uploads
      - datalid-results:/app/results
      - datalid-logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
      - MODEL_DEVICE=0
    env_file:
      - .env.production
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Nginx reverse proxy (opcional)
  nginx:
    image: nginx:alpine
    container_name: datalid-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api

  # Prometheus (opcional)
  prometheus:
    image: prom/prometheus:latest
    container_name: datalid-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

volumes:
  datalid-uploads:
  datalid-results:
  datalid-logs:
  prometheus-data:
```

### Iniciar

```bash
docker-compose up -d
```

## 🌐 Nginx Reverse Proxy

### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name seu-dominio.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name seu-dominio.com;

        # SSL
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # Max upload size
        client_max_body_size 20M;

        # Proxy settings
        location / {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # WebSocket
        location /v2/ws {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        # Health check (não requer auth)
        location /health {
            proxy_pass http://api;
            access_log off;
        }
    }
}
```

## ☁️ Deploy em Cloud

### AWS EC2

```bash
# 1. Launch EC2 instance (p3.2xlarge para GPU)
# 2. SSH into instance
ssh -i key.pem ubuntu@your-instance-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Install NVIDIA Docker (se GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 5. Clone repo
git clone <your-repo>
cd datalid3.0

# 6. Configure
cp .env.example .env.production
nano .env.production  # Edit configs

# 7. Run
docker-compose up -d

# 8. Check logs
docker-compose logs -f api
```

### Google Cloud Run

```bash
# 1. Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/datalid-api

# 2. Deploy
gcloud run deploy datalid-api \
  --image gcr.io/PROJECT_ID/datalid-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars="ENVIRONMENT=production,DEBUG=false"
```

### Azure Container Instances

```bash
# 1. Create resource group
az group create --name datalid --location eastus

# 2. Build and push
az acr build --registry myregistry --image datalid-api:latest .

# 3. Deploy
az container create \
  --resource-group datalid \
  --name datalid-api \
  --image myregistry.azurecr.io/datalid-api:latest \
  --cpu 4 --memory 16 \
  --ports 8000 \
  --environment-variables \
    ENVIRONMENT=production \
    DEBUG=false
```

## 📊 Monitoramento

### Prometheus (prometheus.yml)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'datalid-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/v2/metrics/prometheus'
```

### Grafana Dashboard

1. Add Prometheus as data source
2. Create dashboard with:
   - Request rate
   - Error rate
   - Processing time
   - Images processed
   - Dates found
   - System metrics

### Alerting

```yaml
# alerts.yml
groups:
  - name: datalid_api
    rules:
      - alert: HighErrorRate
        expr: datalid_error_rate_percent > 5
        for: 5m
        annotations:
          summary: "High error rate detected"
      
      - alert: SlowProcessing
        expr: datalid_processing_time_seconds > 5
        for: 10m
        annotations:
          summary: "Processing is slow"
      
      - alert: APIDown
        expr: up{job="datalid-api"} == 0
        for: 1m
        annotations:
          summary: "API is down"
```

## 🔍 Logging

### Centralized Logging

```yaml
# docker-compose.yml (adicionar)
  fluentd:
    image: fluent/fluentd:v1.14
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - datalid-logs:/var/log/datalid
    ports:
      - "24224:24224"

# No serviço api, adicionar:
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: datalid.api
```

### ELK Stack (Elasticsearch, Logstash, Kibana)

```bash
# Enviar logs para Elasticsearch
# Configurar em fluentd.conf ou usar filebeat
```

## 🔐 Segurança

### SSL/TLS

```bash
# Usando Let's Encrypt
sudo apt-get install certbot
sudo certbot certonly --standalone -d seu-dominio.com

# Copiar certificados para nginx
sudo cp /etc/letsencrypt/live/seu-dominio.com/fullchain.pem ./ssl/cert.pem
sudo cp /etc/letsencrypt/live/seu-dominio.com/privkey.pem ./ssl/key.pem
```

### Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Security Best Practices

1. ✅ Nunca exponha portas desnecessárias
2. ✅ Use HTTPS em produção
3. ✅ Implemente autenticação
4. ✅ Rate limiting configurado
5. ✅ Validação de inputs
6. ✅ Logs de audit
7. ✅ Atualizações de segurança regulares
8. ✅ Backup de dados

## 🔄 CI/CD

### GitHub Actions (.github/workflows/deploy.yml)

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t datalid-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push datalid-api:${{ github.sha }}
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            cd /opt/datalid
            docker-compose pull
            docker-compose up -d
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
  api:
    # ... config ...
    deploy:
      replicas: 3  # Múltiplas instâncias
```

### Load Balancer

```nginx
# nginx.conf
upstream api {
    least_conn;  # ou ip_hash
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: datalid-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: datalid-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🧪 Testes em Produção

```bash
# Health check
curl https://seu-dominio.com/health

# Load test (usando hey)
hey -n 1000 -c 10 \
  -m POST \
  -H "X-API-Key: your-key" \
  -D image.jpg \
  https://seu-dominio.com/process

# Monitorar métricas
curl https://seu-dominio.com/v2/metrics
```

## 📞 Suporte

- Logs: `docker-compose logs -f api`
- Métricas: `http://seu-dominio.com/v2/metrics`
- Health: `http://seu-dominio.com/health`

---

## 🎯 Checklist Final

- [ ] Deploy realizado
- [ ] HTTPS configurado
- [ ] Monitoramento funcionando
- [ ] Logs centralizados
- [ ] Backups configurados
- [ ] Alertas configurados
- [ ] Documentação atualizada
- [ ] Equipe treinada

**🎉 Parabéns! API em produção!**
