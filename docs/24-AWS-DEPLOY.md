# 🚀 Deploy na AWS - Guia Completo

Guia passo a passo para fazer deploy da API Datalid na AWS EC2.

## 📋 Sumário

- [Pré-requisitos](#pré-requisitos)
- [Opção 1: EC2 com Docker](#opção-1-ec2-com-docker-recomendado)
- [Opção 2: ECS (Fargate)](#opção-2-ecs-fargate)
- [Opção 3: Lambda (Serverless)](#opção-3-lambda-serverless)
- [Configuração de Domínio](#configuração-de-domínio)
- [SSL/HTTPS](#sslhttps)
- [Monitoramento](#monitoramento)

---

## 📦 Pré-requisitos

### 1. Conta AWS
- Conta AWS ativa
- Acesso ao console AWS
- AWS CLI instalado (opcional)

### 2. Modelo YOLO
```bash
# O modelo está em models/yolov8m-seg.pt
# Tamanho: ~52MB

# Se não tiver, baixe:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -O models/yolov8m-seg.pt
```

### 3. Requisitos da EC2
- **Instância mínima**: `t3.medium` (2 vCPU, 4GB RAM)
- **Recomendado**: `t3.large` ou `t3.xlarge` (com GPU: `g4dn.xlarge`)
- **Storage**: 20GB EBS (mínimo)
- **Sistema**: Amazon Linux 2 ou Ubuntu 22.04

---

## 🎯 Opção 1: EC2 com Docker (Recomendado)

### Passo 1: Criar e Configurar EC2

#### 1.1. Criar instância
```
1. Acesse AWS Console > EC2 > Launch Instance
2. Escolha:
   - Nome: datalid-api-prod
   - AMI: Ubuntu Server 22.04 LTS
   - Tipo: t3.medium (mínimo)
   - Key pair: criar ou usar existente
   - Storage: 20GB gp3
3. Configure Security Group:
   - SSH (22): Seu IP
   - HTTP (80): 0.0.0.0/0
   - HTTPS (443): 0.0.0.0/0
   - Custom TCP (8000): 0.0.0.0/0
```

#### 1.2. Conectar à instância
```bash
# Baixar sua key pair (.pem)
chmod 400 sua-chave.pem

# Conectar via SSH
ssh -i sua-chave.pem ubuntu@SEU-IP-PUBLICO
```

### Passo 2: Instalar Docker

```bash
# Atualizar sistema
sudo apt-get update
sudo apt-get upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER

# Instalar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reiniciar sessão
exit
# Conecte novamente
```

### Passo 3: Fazer Deploy

#### 3.1. Opção A: Via Git (Recomendado)

```bash
# Instalar git
sudo apt-get install -y git

# Clonar repositório
git clone https://github.com/seu-usuario/datalid3.0.git
cd datalid3.0

# Verificar/baixar modelo
ls -lh models/yolov8m-seg.pt

# Se não existir:
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -O models/yolov8m-seg.pt

# Executar script de deploy
chmod +x deploy-aws.sh
./deploy-aws.sh
```

#### 3.2. Opção B: Upload Manual

```bash
# Na sua máquina local, criar arquivo tar
tar -czf datalid-app.tar.gz \
    src/ \
    config/ \
    models/ \
    requirements.txt \
    Dockerfile.production \
    docker-compose.prod.yml \
    deploy-aws.sh

# Upload via SCP
scp -i sua-chave.pem datalid-app.tar.gz ubuntu@SEU-IP:/home/ubuntu/

# Na EC2
tar -xzf datalid-app.tar.gz
cd datalid-app

# Deploy
chmod +x deploy-aws.sh
./deploy-aws.sh
```

### Passo 4: Verificar

```bash
# Ver logs
docker logs -f datalid-api-prod

# Testar API
curl http://localhost:8000/health

# Testar do seu navegador
curl http://SEU-IP-PUBLICO:8000/health
```

### Passo 5: Configurar com Nginx (Opcional mas Recomendado)

```bash
# Iniciar com Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Verificar
docker ps
curl http://localhost/health
```

---

## 🐳 Opção 2: ECS (Fargate)

Para aplicações com escala automática.

### Passo 1: Criar ECR Repository

```bash
# AWS CLI
aws ecr create-repository --repository-name datalid-api --region us-east-1

# Fazer login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin SEU-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com
```

### Passo 2: Build e Push

```bash
# Build
docker build -f Dockerfile.production -t datalid-api:latest .

# Tag
docker tag datalid-api:latest SEU-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com/datalid-api:latest

# Push
docker push SEU-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com/datalid-api:latest
```

### Passo 3: Criar Task Definition

```json
{
  "family": "datalid-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "datalid-api",
    "image": "SEU-ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com/datalid-api:latest",
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "API_HOST", "value": "0.0.0.0"},
      {"name": "API_PORT", "value": "8000"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/datalid-api",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

### Passo 4: Criar Cluster e Service

```bash
# Via AWS Console:
# 1. ECS > Clusters > Create Cluster (Fargate)
# 2. Create Service
# 3. Configure Load Balancer (ALB)
# 4. Configure Auto Scaling
```

---

## ⚡ Opção 3: Lambda (Serverless)

**⚠️ Limitações:**
- Timeout máximo: 15 minutos
- Memória máxima: 10GB
- Tamanho da imagem: 10GB
- **NÃO recomendado** para este caso (modelo pesado)

---

## 🌐 Configuração de Domínio

### Usar Route 53

```
1. Registrar domínio ou importar
2. Criar Hosted Zone
3. Adicionar Record Set:
   - Type: A
   - Name: api.seu-dominio.com
   - Value: IP da EC2 ou ALB
```

### Atualizar Nginx

```nginx
server {
    listen 80;
    server_name api.seu-dominio.com;
    # ... resto da configuração
}
```

---

## 🔒 SSL/HTTPS

### Opção 1: Let's Encrypt (Grátis)

```bash
# Instalar Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Obter certificado
sudo certbot --nginx -d api.seu-dominio.com

# Renovação automática
sudo certbot renew --dry-run
```

### Opção 2: AWS Certificate Manager (ACM)

```
1. ACM > Request Certificate
2. Domain: api.seu-dominio.com
3. Validation: DNS
4. Attach ao ALB (se usar ECS)
```

---

## 📊 Monitoramento

### CloudWatch

```bash
# Instalar CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configurar
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### Logs da API

```bash
# Ver logs em tempo real
docker logs -f datalid-api-prod

# Ver últimas 100 linhas
docker logs --tail 100 datalid-api-prod

# Salvar logs
docker logs datalid-api-prod > logs/api.log
```

### Métricas

```bash
# CPU e Memória do container
docker stats datalid-api-prod

# Disco
df -h

# Memória do sistema
free -h
```

---

## 💰 Estimativa de Custos (us-east-1)

### Opção 1: EC2 t3.medium
- **Instância**: $0.0416/hora = ~$30/mês
- **Storage**: 20GB EBS = ~$2/mês
- **Data Transfer**: Primeiros 100GB grátis
- **Total**: ~$32/mês

### Opção 2: EC2 t3.large
- **Instância**: $0.0832/hora = ~$60/mês
- **Storage**: 20GB EBS = ~$2/mês
- **Total**: ~$62/mês

### Opção 3: ECS Fargate
- **vCPU**: 2 vCPU = $0.04048/hora
- **Memória**: 4GB = $0.004445/hora
- **Total**: ~$32/mês (24/7)

### Opção 4: Com GPU (g4dn.xlarge)
- **Instância**: $0.526/hora = ~$380/mês
- Recomendado apenas para alta demanda

---

## 🔧 Comandos Úteis

### Docker

```bash
# Ver containers
docker ps

# Ver logs
docker logs -f datalid-api-prod

# Acessar container
docker exec -it datalid-api-prod bash

# Reiniciar
docker restart datalid-api-prod

# Parar
docker stop datalid-api-prod

# Remover
docker rm -f datalid-api-prod

# Ver uso de recursos
docker stats

# Limpar imagens antigas
docker system prune -a
```

### Sistema

```bash
# CPU e Memória
htop

# Disco
df -h

# Rede
netstat -tulpn | grep 8000

# Processos
ps aux | grep python
```

---

## 🐛 Troubleshooting

### API não inicia

```bash
# Ver logs
docker logs datalid-api-prod

# Verificar porta
sudo lsof -i :8000

# Reiniciar
docker restart datalid-api-prod
```

### Out of Memory

```bash
# Ver memória
free -h
docker stats

# Solução: Aumentar instância ou adicionar swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Modelo não encontrado

```bash
# Verificar
docker exec datalid-api-prod ls -lh /app/models/

# Baixar manualmente
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -O models/yolov8m-seg.pt

# Rebuild
docker build -f Dockerfile.production -t datalid-api:latest .
```

---

## ✅ Checklist de Deploy

- [ ] EC2 criada e configurada
- [ ] Security Group configurado (portas 22, 80, 443, 8000)
- [ ] Docker instalado
- [ ] Modelo YOLO baixado
- [ ] Código copiado para EC2
- [ ] Docker image construída
- [ ] Container rodando
- [ ] Health check OK
- [ ] API acessível externamente
- [ ] (Opcional) Domínio configurado
- [ ] (Opcional) SSL configurado
- [ ] (Opcional) Monitoring configurado

---

## 📚 Recursos

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/)

---

**Última atualização:** 16/11/2025
