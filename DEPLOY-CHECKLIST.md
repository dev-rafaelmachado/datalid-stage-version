# ✅ Checklist de Deploy AWS

Use este checklist para garantir que tudo está configurado corretamente.

## 📋 Pré-Deploy

### AWS Account
- [ ] Conta AWS ativa
- [ ] Cartão de crédito configurado
- [ ] AWS CLI instalado (opcional)
- [ ] IAM user com permissões EC2

### Recursos Locais
- [ ] Código no GitHub/GitLab (recomendado)
- [ ] Modelo YOLO baixado (`models/yolov8m-seg.pt` - 52MB)
- [ ] Arquivo `.env.production` configurado
- [ ] Scripts testados localmente

## 🖥️ Configuração EC2

### Instância
- [ ] Tipo: `t3.medium` ou superior
- [ ] AMI: Ubuntu 22.04 LTS
- [ ] vCPUs: 2+
- [ ] RAM: 4GB+
- [ ] Storage: 20GB+ (EBS gp3)
- [ ] Key pair (.pem) baixado e salvo
- [ ] Elastic IP associado (opcional mas recomendado)

### Security Group
- [ ] SSH (22): Seu IP ou 0.0.0.0/0
- [ ] HTTP (80): 0.0.0.0/0
- [ ] HTTPS (443): 0.0.0.0/0
- [ ] Custom TCP (8000): 0.0.0.0/0

### Network
- [ ] VPC padrão ou custom configurada
- [ ] Subnet pública selecionada
- [ ] Auto-assign Public IP: Enabled

## 🐳 Software na EC2

### Sistema Base
- [ ] Ubuntu atualizado (`sudo apt-get update && upgrade`)
- [ ] Git instalado (`sudo apt-get install git`)
- [ ] Curl/Wget instalado

### Docker
- [ ] Docker Engine instalado
- [ ] Docker Compose instalado
- [ ] Usuário adicionado ao grupo docker
- [ ] Docker rodando (`docker ps`)

## 📦 Deploy da Aplicação

### Código
- [ ] Repositório clonado ou arquivos copiados
- [ ] Estrutura de diretórios OK
- [ ] Permissões corretas (`chmod +x deploy-aws.sh`)

### Modelo YOLO
- [ ] Arquivo existe: `models/yolov8m-seg.pt`
- [ ] Tamanho correto: ~52MB
- [ ] Permissões de leitura OK

### Build Docker
- [ ] Dockerfile.production existe
- [ ] Build executado sem erros
- [ ] Imagem criada (`docker images`)
- [ ] Tamanho da imagem: ~2-3GB

### Container
- [ ] Container iniciado
- [ ] Porta 8000 exposta
- [ ] Volumes montados (uploads, outputs, logs)
- [ ] Variáveis de ambiente configuradas

## 🧪 Testes

### Health Check
- [ ] `/health` responde localmente (localhost:8000)
- [ ] `/health` responde externamente (IP-PUBLICO:8000)
- [ ] Status: "healthy"

### API Endpoints
- [ ] `/` (root) responde
- [ ] `/docs` acessível
- [ ] `/openapi.json` acessível

### Funcional
- [ ] Upload de imagem funciona
- [ ] OCR extrai texto
- [ ] Datas são parseadas
- [ ] Response JSON válido

### Performance
- [ ] Tempo de resposta < 5s (imagem típica)
- [ ] Uso de CPU < 80%
- [ ] Uso de memória < 80%
- [ ] Sem memory leaks

## 🔒 Segurança

### Acesso
- [ ] Senha SSH forte ou key-based
- [ ] Porta SSH mudada (opcional)
- [ ] Fail2ban instalado (opcional)
- [ ] Usuário não-root criado

### Firewall
- [ ] Security Group configurado
- [ ] UFW configurado (opcional)
- [ ] Rate limiting no Nginx (se usar)

### SSL/HTTPS
- [ ] Certificado SSL instalado (Let's Encrypt ou ACM)
- [ ] Redirecionamento HTTP → HTTPS
- [ ] HSTS header configurado

## 🌐 Domínio (Opcional)

### DNS
- [ ] Domínio registrado
- [ ] Route 53 ou outro DNS configurado
- [ ] Record A apontando para IP da EC2
- [ ] TTL configurado (300s recomendado)

### Nginx
- [ ] Server name configurado
- [ ] Virtual host configurado
- [ ] Proxy pass para API

## 📊 Monitoramento

### Logs
- [ ] Logs da API acessíveis (`docker logs`)
- [ ] Rotação de logs configurada
- [ ] Logs persistentes (volume montado)

### Métricas
- [ ] CloudWatch configurado (opcional)
- [ ] Alarmes configurados (CPU, Memória)
- [ ] Dashboard criado

### Alertas
- [ ] Email/SMS configurado para alarmes
- [ ] Slack/Discord webhook (opcional)

## 🔄 Automação

### CI/CD (Opcional)
- [ ] GitHub Actions configurado
- [ ] Deploy automático em push
- [ ] Testes automatizados

### Backups
- [ ] Snapshots EBS configurados
- [ ] Frequência definida (diário recomendado)
- [ ] Retenção definida (7 dias recomendado)

### Updates
- [ ] Sistema de atualização definido
- [ ] Downtime mínimo planejado
- [ ] Rollback strategy definida

## 💰 Custos

### Recursos
- [ ] Instância EC2
- [ ] Storage EBS
- [ ] Data Transfer
- [ ] Elastic IP (se usar)
- [ ] Route 53 (se usar)

### Otimização
- [ ] Reserved Instance considerada (se uso constante)
- [ ] Savings Plans considerado
- [ ] Spot Instance considerada (se workload permite)

### Limites
- [ ] Billing alerts configurados
- [ ] Budget definido
- [ ] Tags de custo aplicadas

## 🚀 Frontend Integration

### Configuração
- [ ] URL da API configurada no frontend
- [ ] CORS configurado na API
- [ ] Headers corretos nas requests
- [ ] Error handling implementado

### Testes E2E
- [ ] Upload de imagem do frontend funciona
- [ ] Resposta da API é processada
- [ ] UI atualiza corretamente
- [ ] Loading states funcionam

## 📝 Documentação

### Equipe
- [ ] README atualizado
- [ ] URLs documentadas
- [ ] Credenciais salvas (cofre seguro)
- [ ] Runbook criado

### Handoff
- [ ] Acesso compartilhado (se necessário)
- [ ] Procedimentos documentados
- [ ] Contatos de suporte definidos

## 🎯 Go-Live Final

### Última Verificação
- [ ] Todos os testes passando
- [ ] Performance aceitável
- [ ] Sem erros nos logs
- [ ] Backup recente criado

### Comunicação
- [ ] Stakeholders informados
- [ ] Equipe alinhada
- [ ] Suporte preparado

### Launch
- [ ] DNS atualizado (se usar domínio)
- [ ] Frontend apontando para nova API
- [ ] Monitoramento ativo
- [ ] On-call definido

---

## 🎉 Parabéns!

Se todos os itens estão marcados, sua API está em produção! 🚀

### Próximos Passos

1. **Monitorar** primeiras horas/dias
2. **Coletar métricas** de uso
3. **Otimizar** conforme necessário
4. **Escalar** se precisar

### Recursos Úteis

- [Documentação Completa](docs/24-AWS-DEPLOY.md)
- [Deploy Rápido](DEPLOY-QUICK.md)
- [API Docs](http://SEU-IP:8000/docs)

---

**Data do Deploy:** _______________

**Responsável:** _______________

**IP da EC2:** _______________

**URL da API:** _______________
