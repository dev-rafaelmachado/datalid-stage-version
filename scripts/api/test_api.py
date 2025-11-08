"""
🧪 Script de Teste da API
Testa todos os endpoints da API Datalid.
"""

import json
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

console = Console()

# Configuração
API_URL = "http://localhost:8000"
TEST_IMAGE = "data/sample.jpg"


def print_header(title: str):
    """Imprime header formatado."""
    console.print(f"\n{'=' * 80}", style="bold blue")
    console.print(f"{title:^80}", style="bold blue")
    console.print(f"{'=' * 80}\n", style="bold blue")


def test_root():
    """Testa endpoint root."""
    print_header("Teste: Root Endpoint")
    
    response = requests.get(f"{API_URL}/")
    
    console.print(f"Status: {response.status_code}", style="green" if response.status_code == 200 else "red")
    console.print(f"Response: {json.dumps(response.json(), indent=2)}", style="cyan")


def test_health():
    """Testa health check."""
    print_header("Teste: Health Check")
    
    response = requests.get(f"{API_URL}/health")
    
    console.print(f"Status: {response.status_code}", style="green" if response.status_code == 200 else "red")
    
    data = response.json()
    
    # Mostrar status geral
    console.print(f"\nStatus Geral: {data['status']}", style="bold green")
    console.print(f"Versão: {data['version']}")
    
    # Mostrar componentes
    console.print("\nComponentes:", style="bold")
    for name, comp in data['components'].items():
        status_color = "green" if comp['status'] == "healthy" else "red"
        console.print(f"  • {name}: {comp['status']}", style=status_color)
        if comp.get('message'):
            console.print(f"    {comp['message']}", style="dim")


def test_info():
    """Testa endpoint de info."""
    print_header("Teste: API Info")
    
    response = requests.get(f"{API_URL}/info")
    
    console.print(f"Status: {response.status_code}", style="green" if response.status_code == 200 else "red")
    
    data = response.json()
    
    console.print(f"\nNome: {data['name']}")
    console.print(f"Versão: {data['version']}")
    console.print(f"Descrição: {data['description']}")
    
    console.print(f"\nOCR Engines: {', '.join(data['ocr_engines'])}", style="cyan")
    
    console.print("\nLimites:", style="bold")
    for key, value in data['limits'].items():
        console.print(f"  • {key}: {value}")


def test_process_image():
    """Testa processamento de imagem."""
    print_header("Teste: Processar Imagem")
    
    if not Path(TEST_IMAGE).exists():
        console.print(f"❌ Imagem de teste não encontrada: {TEST_IMAGE}", style="red")
        return
    
    console.print(f"📷 Enviando imagem: {TEST_IMAGE}", style="cyan")
    
    start_time = time.time()
    
    with open(TEST_IMAGE, 'rb') as f:
        files = {'file': f}
        data = {
            'detection_confidence': 0.25,
            'detection_iou': 0.7,
            'return_visualization': False,
            'return_crops': False,
            'return_full_ocr': True,
        }
        
        response = requests.post(
            f"{API_URL}/process",
            files=files,
            data=data
        )
    
    elapsed = time.time() - start_time
    
    console.print(f"\n✓ Status: {response.status_code}", style="green" if response.status_code == 200 else "red")
    console.print(f"✓ Tempo de resposta: {elapsed:.2f}s", style="cyan")
    
    if response.status_code == 200:
        result = response.json()
        
        console.print(f"\n📊 Resultados:", style="bold")
        console.print(f"  • Status: {result['status']}", style="green")
        console.print(f"  • Mensagem: {result['message']}")
        console.print(f"  • Detecções: {len(result['detections'])}")
        console.print(f"  • Datas encontradas: {len(result['dates'])}")
        
        if result['best_date']:
            console.print(f"\n📅 Melhor Data:", style="bold green")
            best = result['best_date']
            console.print(f"  • Data: {best['date']}")
            console.print(f"  • Confiança: {best['confidence']:.2%}")
            console.print(f"  • Formato: {best['format']}")
            console.print(f"  • Válida: {best['is_valid']}")
            if best['is_expired'] is not None:
                status = "Expirada" if best['is_expired'] else "Válida"
                console.print(f"  • Status: {status}")
            if best['days_until_expiry'] is not None:
                console.print(f"  • Dias até expirar: {best['days_until_expiry']}")
        
        console.print(f"\n⚡ Métricas:", style="bold")
        metrics = result['metrics']
        console.print(f"  • Tempo total: {metrics['total_time']:.3f}s")
        console.print(f"  • Detecção: {metrics['detection_time']:.3f}s")
        console.print(f"  • OCR: {metrics['ocr_time']:.3f}s")
        console.print(f"  • Parsing: {metrics['parsing_time']:.3f}s")
        
        if result.get('ocr_results'):
            console.print(f"\n📝 Resultados OCR:", style="bold")
            for i, ocr in enumerate(result['ocr_results'], 1):
                console.print(f"  [{i}] Texto: '{ocr['text']}'")
                console.print(f"      Confiança: {ocr['confidence']:.2%}")
                console.print(f"      Engine: {ocr['engine']}")
    else:
        console.print(f"\n❌ Erro: {response.text}", style="red")


def test_process_batch():
    """Testa processamento em batch."""
    print_header("Teste: Processar Batch")
    
    # Procurar imagens de teste
    test_dir = Path("data/tests")
    if not test_dir.exists():
        console.print(f"❌ Diretório de teste não encontrado: {test_dir}", style="red")
        return
    
    images = list(test_dir.glob("*.jpg"))[:3]  # Máximo 3 para teste
    
    if not images:
        console.print(f"❌ Nenhuma imagem encontrada em: {test_dir}", style="red")
        return
    
    console.print(f"📷 Enviando {len(images)} imagens", style="cyan")
    
    start_time = time.time()
    
    files = [('files', open(img, 'rb')) for img in images]
    
    try:
        response = requests.post(
            f"{API_URL}/process/batch",
            files=files
        )
    finally:
        for _, f in files:
            f.close()
    
    elapsed = time.time() - start_time
    
    console.print(f"\n✓ Status: {response.status_code}", style="green" if response.status_code == 200 else "red")
    console.print(f"✓ Tempo de resposta: {elapsed:.2f}s", style="cyan")
    
    if response.status_code == 200:
        result = response.json()
        
        console.print(f"\n📊 Resultados do Batch:", style="bold")
        console.print(f"  • Status: {result['status']}", style="green")
        console.print(f"  • Total: {result['total_images']}")
        console.print(f"  • Sucesso: {result['successful']}", style="green")
        console.print(f"  • Falhas: {result['failed']}", style="red" if result['failed'] > 0 else "dim")
        console.print(f"  • Tempo total: {result['total_time']:.2f}s")
        
        # Tabela de resultados
        table = Table(title="Resultados por Imagem")
        table.add_column("Imagem", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Datas", style="yellow")
        table.add_column("Melhor Data", style="magenta")
        
        for item in result['results']:
            status = "✓" if item['success'] else "✗"
            
            if item['success'] and item['result']:
                num_dates = len(item['result']['dates'])
                best_date = item['result']['best_date']['date'] if item['result']['best_date'] else "N/A"
            else:
                num_dates = 0
                best_date = "N/A"
            
            table.add_row(
                item['filename'],
                status,
                str(num_dates),
                best_date
            )
        
        console.print(table)
    else:
        console.print(f"\n❌ Erro: {response.text}", style="red")


def test_invalid_file():
    """Testa envio de arquivo inválido."""
    print_header("Teste: Arquivo Inválido")
    
    # Criar arquivo texto temporário
    content = b"Este nao e uma imagem"
    
    files = {'file': ('test.txt', content, 'text/plain')}
    
    response = requests.post(
        f"{API_URL}/process",
        files=files
    )
    
    console.print(f"Status: {response.status_code}", style="yellow")
    console.print(f"Response: {json.dumps(response.json(), indent=2)}", style="cyan")
    
    if response.status_code == 400:
        console.print("\n✓ Validação funcionando corretamente!", style="green")
    else:
        console.print("\n✗ Validação não funcionou como esperado", style="red")


def run_all_tests():
    """Executa todos os testes."""
    console.print("\n" + "🧪 INICIANDO TESTES DA API ".center(80, "="), style="bold magenta")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("API Info", test_info),
        ("Processar Imagem", test_process_image),
        ("Processar Batch", test_process_batch),
        ("Arquivo Inválido", test_invalid_file),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
            console.print(f"\n✓ {name} - PASSOU", style="bold green")
        except Exception as e:
            results.append((name, False, str(e)))
            console.print(f"\n✗ {name} - FALHOU: {e}", style="bold red")
        
        time.sleep(1)  # Pequena pausa entre testes
    
    # Resumo
    print_header("Resumo dos Testes")
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    console.print(f"\nTotal de testes: {len(results)}")
    console.print(f"Passou: {passed}", style="green")
    console.print(f"Falhou: {failed}", style="red" if failed > 0 else "dim")
    console.print(f"Taxa de sucesso: {passed/len(results)*100:.1f}%", style="cyan")
    
    # Detalhes de falhas
    failures = [(name, error) for name, success, error in results if not success]
    if failures:
        console.print("\n❌ Testes que falharam:", style="bold red")
        for name, error in failures:
            console.print(f"  • {name}: {error}", style="red")


if __name__ == "__main__":
    import sys

    # Verificar se API está rodando
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
    except requests.exceptions.ConnectionError:
        console.print(f"\n❌ API não está rodando em {API_URL}", style="bold red")
        console.print("\nInicie a API primeiro:", style="yellow")
        console.print("  python -m src.api.main", style="cyan")
        sys.exit(1)
    
    run_all_tests()
    
    console.print("\n" + "=" * 80, style="bold blue")
    console.print("✅ Testes concluídos!", style="bold green")
    console.print("=" * 80 + "\n", style="bold blue")
