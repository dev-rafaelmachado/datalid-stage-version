#!/bin/bash

# ========================================
# 🚀 Script de Deploy - AWS EC2
# ========================================

set -e  # Parar em caso de erro

echo "========================================="
echo "🚀 Datalid API - Deploy AWS EC2"
echo "========================================="

# Variáveis
APP_NAME="datalid-api"
DOCKER_IMAGE="datalid-api:latest"
CONTAINER_NAME="datalid-api-prod"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funções auxiliares
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ========================================
# 1. Verificar requisitos
# ========================================
print_step "Verificando requisitos..."

if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    echo "Instale o Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_warning "Docker Compose não encontrado. Tentando instalar..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

print_step "✅ Requisitos OK"

# ========================================
# 2. Parar containers antigos
# ========================================
print_step "Parando containers antigos..."

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    docker stop $CONTAINER_NAME
    print_step "Container parado"
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    docker rm $CONTAINER_NAME
    print_step "Container removido"
fi

# ========================================
# 3. Build da imagem
# ========================================
print_step "Construindo imagem Docker..."

docker build -f Dockerfile.production -t $DOCKER_IMAGE .

print_step "✅ Imagem construída: $DOCKER_IMAGE"

# ========================================
# 4. Verificar modelo YOLO
# ========================================
print_step "Verificando modelo YOLO..."

if [ ! -f "models/yolov8m-seg.pt" ]; then
    print_warning "Modelo não encontrado em models/yolov8m-seg.pt"
    print_warning "Baixando modelo..."
    
    mkdir -p models
    
    # Opção 1: Usar wget
    if command -v wget &> /dev/null; then
        wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -O models/yolov8m-seg.pt
    # Opção 2: Usar curl
    elif command -v curl &> /dev/null; then
        curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -o models/yolov8m-seg.pt
    else
        print_error "wget ou curl não encontrado. Baixe manualmente o modelo."
        exit 1
    fi
    
    print_step "✅ Modelo baixado"
else
    print_step "✅ Modelo encontrado"
fi

# ========================================
# 5. Criar diretórios necessários
# ========================================
print_step "Criando diretórios..."

mkdir -p uploads outputs logs temp

print_step "✅ Diretórios criados"

# ========================================
# 6. Iniciar container
# ========================================
print_step "Iniciando container..."

docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p 8000:8000 \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/logs:/app/logs \
    -e API_HOST=0.0.0.0 \
    -e API_PORT=8000 \
    -e LOG_LEVEL=INFO \
    $DOCKER_IMAGE

print_step "✅ Container iniciado"

# ========================================
# 7. Aguardar inicialização
# ========================================
print_step "Aguardando inicialização (30s)..."

sleep 30

# ========================================
# 8. Health check
# ========================================
print_step "Verificando saúde da API..."

MAX_RETRIES=5
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_step "✅ API está saudável!"
        break
    else
        RETRY=$((RETRY+1))
        print_warning "Tentativa $RETRY/$MAX_RETRIES..."
        sleep 5
    fi
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    print_error "API não respondeu ao health check"
    print_error "Logs do container:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# ========================================
# 9. Teste básico
# ========================================
print_step "Executando teste básico..."

API_INFO=$(curl -s http://localhost:8000/)

if [ $? -eq 0 ]; then
    print_step "✅ API respondendo corretamente"
    echo "$API_INFO" | python3 -m json.tool || echo "$API_INFO"
else
    print_error "API não está respondendo"
    exit 1
fi

# ========================================
# 10. Informações finais
# ========================================
echo ""
echo "========================================="
echo "✅ DEPLOY CONCLUÍDO COM SUCESSO!"
echo "========================================="
echo ""
echo "📊 Informações:"
echo "  - Container: $CONTAINER_NAME"
echo "  - Imagem: $DOCKER_IMAGE"
echo "  - Porta: 8000"
echo ""
echo "🔗 URLs:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "📝 Comandos úteis:"
echo "  - Ver logs: docker logs -f $CONTAINER_NAME"
echo "  - Parar: docker stop $CONTAINER_NAME"
echo "  - Reiniciar: docker restart $CONTAINER_NAME"
echo "  - Remover: docker rm -f $CONTAINER_NAME"
echo ""
echo "🎯 Próximos passos:"
echo "  1. Teste a API com: curl http://localhost:8000/health"
echo "  2. Configure o Security Group da EC2 para permitir porta 8000"
echo "  3. Configure um domínio (opcional)"
echo "  4. Configure SSL/HTTPS (opcional)"
echo ""
