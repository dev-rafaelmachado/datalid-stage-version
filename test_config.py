#!/usr/bin/env python3
"""
🧪 Testar Configurações da API

Verifica se as variáveis de ambiente estão sendo carregadas corretamente.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.config import get_settings


def test_config():
    """Testa carregamento das configurações."""
    print("=" * 70)
    print("🧪 Testando Configurações da API")
    print("=" * 70)
    
    # Carregar configurações
    settings = get_settings()
    
    print("\n📋 CONFIGURAÇÕES CARREGADAS:\n")
    
    # Servidor
    print("🖥️  SERVIDOR:")
    print(f"   Host: {settings.host}")
    print(f"   Port: {settings.port}")
    print(f"   Workers: {settings.workers}")
    print(f"   Environment: {settings.environment}")
    print(f"   Debug: {settings.debug}")
    
    # YOLO
    print("\n🎯 MODELO YOLO:")
    print(f"   Caminho configurado: {settings.yolo_model_path}")
    
    try:
        model_path = settings.get_yolo_model_path()
        print(f"   Caminho resolvido: {model_path}")
        print(f"   Existe: {'✅ SIM' if model_path.exists() else '❌ NÃO'}")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"   Tamanho: {size_mb:.1f} MB")
    except Exception as e:
        print(f"   ❌ Erro ao resolver caminho: {e}")
    
    print(f"   Device: {settings.get_yolo_device()}")
    print(f"   Use GPU: {settings.use_gpu}")
    print(f"   Confidence: {settings.default_confidence}")
    print(f"   IoU: {settings.default_iou}")
    
    # OCR
    print("\n📝 OCR:")
    print(f"   Engine: {settings.default_ocr_engine}")
    print(f"   Config: {settings.default_ocr_config}")
    print(f"   Preprocessing: {settings.default_preprocessing_config}")
    
    # Limites
    print("\n⚙️  LIMITES:")
    print(f"   Max file size: {settings.max_file_size_mb} MB")
    print(f"   Max batch size: {settings.max_batch_size}")
    print(f"   Allowed extensions: {settings.allowed_extensions}")
    print(f"   Request timeout: {settings.request_timeout}s")
    
    # CORS
    print("\n🌐 CORS:")
    print(f"   Enabled: {settings.cors_enabled}")
    if settings.cors_enabled:
        print(f"   Origins: {settings.cors_origins}")
    
    # Storage
    print("\n💾 STORAGE:")
    print(f"   Upload dir: {settings.upload_dir}")
    print(f"   Results dir: {settings.results_dir}")
    print(f"   Cleanup uploads: {settings.cleanup_uploads}")
    
    # Logging
    print("\n📊 LOGGING:")
    print(f"   Level: {settings.log_level}")
    print(f"   Log requests: {settings.log_requests}")
    print(f"   Log file: {settings.log_file}")
    
    # Auth
    print("\n🔒 AUTENTICAÇÃO:")
    print(f"   Enabled: {settings.auth_enabled}")
    
    # Teste de configuração do YOLO
    print("\n" + "=" * 70)
    print("🎯 CONFIGURAÇÃO COMPLETA DO YOLO:")
    print("=" * 70)
    
    yolo_config = settings.get_yolo_config()
    for key, value in yolo_config.items():
        print(f"   {key}: {value}")
    
    # Validações
    print("\n" + "=" * 70)
    print("✅ VALIDAÇÕES:")
    print("=" * 70)
    
    issues = []
    
    # Verificar modelo YOLO
    try:
        model_path = settings.get_yolo_model_path()
        if not model_path.exists():
            issues.append(f"❌ Modelo YOLO não encontrado: {model_path}")
    except Exception as e:
        issues.append(f"❌ Erro ao verificar modelo YOLO: {e}")
    
    # Verificar diretórios
    for dir_name, dir_path in [
        ("Upload", settings.upload_dir),
        ("Results", settings.results_dir),
    ]:
        path = Path(dir_path)
        if not path.exists():
            issues.append(f"⚠️  Diretório {dir_name} não existe: {dir_path}")
    
    # Verificar arquivos de configuração
    for config_name, config_path in [
        ("OCR", settings.default_ocr_config),
        ("Preprocessing", settings.default_preprocessing_config),
    ]:
        path = Path(config_path)
        if not path.exists():
            issues.append(f"⚠️  Config {config_name} não encontrado: {config_path}")
    
    # Mostrar resultados
    if issues:
        print("\n⚠️  PROBLEMAS ENCONTRADOS:\n")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ Tudo OK! Configurações válidas.")
    
    print("\n" + "=" * 70)
    print("🏁 TESTE CONCLUÍDO")
    print("=" * 70)
    
    return len(issues) == 0


def test_env_loading():
    """Testa se variáveis de ambiente estão sendo carregadas."""
    print("\n" + "=" * 70)
    print("🔍 VERIFICANDO VARIÁVEIS DE AMBIENTE")
    print("=" * 70)
    
    import os
    
    env_vars = [
        "API_HOST",
        "API_PORT",
        "API_YOLO_MODEL_PATH",
        "API_MODEL_DEVICE",
        "API_USE_GPU",
        "API_DEFAULT_OCR_ENGINE",
        "API_LOG_LEVEL",
    ]
    
    print("\n📋 Variáveis de ambiente detectadas:\n")
    
    found = 0
    for var in env_vars:
        value = os.getenv(var)
        if value is not None:
            print(f"   ✅ {var} = {value}")
            found += 1
        else:
            print(f"   ❌ {var} = (não definida)")
    
    print(f"\n📊 Encontradas: {found}/{len(env_vars)}")
    
    if found == 0:
        print("\n⚠️  ATENÇÃO: Nenhuma variável de ambiente encontrada!")
        print("   Certifique-se de ter um arquivo .env na raiz do projeto")
        print("   Ou exporte as variáveis manualmente")
    
    return found > 0


def main():
    """Executar todos os testes."""
    print("\n🚀 Iniciando testes de configuração...\n")
    
    # Teste 1: Variáveis de ambiente
    env_ok = test_env_loading()
    
    # Teste 2: Configurações
    config_ok = test_config()
    
    # Resultado final
    print("\n" + "=" * 70)
    print("📊 RESULTADO FINAL")
    print("=" * 70)
    
    if env_ok and config_ok:
        print("✅ SUCESSO: Configurações carregadas corretamente!")
        return 0
    else:
        print("❌ FALHA: Problemas encontrados nas configurações")
        print("\n💡 Dicas:")
        print("   1. Certifique-se de ter um arquivo .env")
        print("   2. Verifique se o modelo YOLO está no local correto")
        print("   3. Ajuste os caminhos conforme seu ambiente")
        return 1


if __name__ == "__main__":
    sys.exit(main())
