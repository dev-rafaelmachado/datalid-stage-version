# 🚀 Deploy Rápido - AWS EC2

## 📋 Pré-requisitos

- Instância EC2 criada (t3.medium ou maior)
- Security Group: portas 22, 80, 8000 abertas
- Ubuntu 22.04 LTS

## ⚡ Deploy em 5 Minutos

### 1. Conectar à EC2

```bash
ssh -i sua-chave.pem ubuntu@SEU-IP-PUBLICO
```

### 2. Instalar Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

**Importante:** Saia e entre novamente no SSH após este comando!

### 3. Clonar e Deploy

```bash
# Clonar repo
git clone https://github.com/seu-usuario/datalid3.0.git
cd datalid3.0

# Verificar modelo (deve ter ~52MB)
ls -lh models/yolov8m-seg.pt

# Se não existir, baixar:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt -O models/yolov8m-seg.pt

# Deploy!
chmod +x deploy-aws.sh
./deploy-aws.sh
```

### 4. Testar

```bash
# Local
curl http://localhost:8000/health

# Externo (troque pelo seu IP)
curl http://SEU-IP-PUBLICO:8000/health
```

## 🎯 URLs Importantes

- **API**: `http://SEU-IP-PUBLICO:8000`
- **Docs**: `http://SEU-IP-PUBLICO:8000/docs`
- **Health**: `http://SEU-IP-PUBLICO:8000/health`

## 📱 Testar no Frontend

```javascript
// Atualizar URL da API no seu frontend
const API_URL = 'http://SEU-IP-PUBLICO:8000';

// Testar
fetch(`${API_URL}/health`)
  .then(r => r.json())
  .then(data => console.log(data));
```

## 🔧 Comandos Úteis

```bash
# Ver logs
docker logs -f datalid-api-prod

# Reiniciar
docker restart datalid-api-prod

# Parar
docker stop datalid-api-prod

# Status
docker ps
docker stats
```

## ⚠️ Importante para Produção

1. **Security Group**: Configure para aceitar apenas do seu domínio frontend
2. **HTTPS**: Configure SSL com Let's Encrypt (veja docs/24-AWS-DEPLOY.md)
3. **Domínio**: Configure um domínio próprio (opcional mas recomendado)
4. **Backups**: Configure backups automáticos do EBS
5. **Monitoring**: Configure CloudWatch (veja docs/24-AWS-DEPLOY.md)

## 📚 Documentação Completa

Veja [docs/24-AWS-DEPLOY.md](docs/24-AWS-DEPLOY.md) para:
- Configuração de domínio
- SSL/HTTPS
- Monitoring
- ECS/Fargate
- Troubleshooting

## 💰 Custos Estimados

- **t3.medium**: ~$30/mês
- **t3.large**: ~$60/mês
- **Data transfer**: Primeiros 100GB grátis

## 🐛 Problemas?

1. API não inicia? `docker logs datalid-api-prod`
2. Modelo não encontrado? Verifique `models/yolov8m-seg.pt`
3. Sem memória? Aumente a instância ou adicione swap
4. Porta bloqueada? Verifique Security Group

## ✅ Checklist

- [ ] EC2 criada (t3.medium+)
- [ ] Security Group configurado
- [ ] Docker instalado
- [ ] Repositório clonado
- [ ] Modelo YOLO baixado (~52MB)
- [ ] Deploy executado (`./deploy-aws.sh`)
- [ ] Health check OK
- [ ] API acessível externamente
- [ ] Frontend conectado

---

**Pronto!** Sua API está no ar! 🎉

Para configurações avançadas, veja a [documentação completa](docs/24-AWS-DEPLOY.md).
