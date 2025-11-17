#!/usr/bin/env python3
"""
🧪 Exemplo: API com Segmentação e Crops

Demonstra o uso dos novos recursos da API:
- Máscaras de segmentação (polígonos)
- Imagens dos crops (original e pré-processado)
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def test_segmentation_only(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Teste 1: Apenas segmentação (sem crops).
    """
    print("=" * 70)
    print("🎭 TESTE 1: Segmentação")
    print("=" * 70)
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/process",
            files={'file': f}
        )
    
    result = response.json()
    
    print(f"\n✅ Status: {result['status']}")
    print(f"📊 Detecções: {result['metrics']['num_detections']}")
    
    for i, detection in enumerate(result['detections'], 1):
        print(f"\n🔍 Detecção {i}:")
        print(f"   Confiança: {detection['confidence']:.2%}")
        print(f"   Tem máscara: {detection['has_mask']}")
        
        if detection.get('segmentation'):
            seg = detection['segmentation']
            print(f"   Pontos do polígono: {len(seg)}")
            print(f"   Primeiro ponto: {seg[0]}")
            print(f"   Último ponto: {seg[-1]}")
    
    for i, ocr in enumerate(result['ocr_results'], 1):
        print(f"\n📝 OCR {i}:")
        print(f"   Texto: {ocr['text']}")
        print(f"   Confiança: {ocr['confidence']:.2%}")
        print(f"   Tem crop original: {'crop_original_base64' in ocr}")
        print(f"   Tem crop processado: {'crop_processed_base64' in ocr}")
    
    return result


def test_with_crop_images(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Teste 2: Com imagens dos crops.
    """
    print("\n" + "=" * 70)
    print("🖼️  TESTE 2: Com Imagens dos Crops")
    print("=" * 70)
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/process",
            files={'file': f},
            data={'return_crop_images': 'true'}
        )
    
    result = response.json()
    
    print(f"\n✅ Status: {result['status']}")
    
    for i, ocr in enumerate(result['ocr_results'], 1):
        print(f"\n📝 OCR {i}:")
        print(f"   Texto: {ocr['text']}")
        print(f"   Confiança: {ocr['confidence']:.2%}")
        
        # Verificar crops
        has_original = ocr.get('crop_original_base64') is not None
        has_processed = ocr.get('crop_processed_base64') is not None
        
        print(f"   ✅ Crop original: {'SIM' if has_original else 'NÃO'}")
        print(f"   ✅ Crop processado: {'SIM' if has_processed else 'NÃO'}")
        
        if has_original:
            # Salvar crop original
            crop_original = decode_base64_image(ocr['crop_original_base64'])
            output_path = f"crop_{i}_original.jpg"
            crop_original.save(output_path)
            print(f"   💾 Salvo: {output_path}")
        
        if has_processed:
            # Salvar crop processado
            crop_processed = decode_base64_image(ocr['crop_processed_base64'])
            output_path = f"crop_{i}_processed.jpg"
            crop_processed.save(output_path)
            print(f"   💾 Salvo: {output_path}")
    
    return result


def test_full_pipeline(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Teste 3: Pipeline completo (segmentação + crops + visualização).
    """
    print("\n" + "=" * 70)
    print("🚀 TESTE 3: Pipeline Completo")
    print("=" * 70)
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/process",
            files={'file': f},
            data={
                'return_crop_images': 'true',
                'return_visualization': 'true'
            }
        )
    
    result = response.json()
    
    print(f"\n✅ Status: {result['status']}")
    print(f"📊 Métricas:")
    print(f"   Tempo total: {result['metrics']['total_time']:.2f}s")
    print(f"   Detecções: {result['metrics']['num_detections']}")
    print(f"   Datas encontradas: {result['metrics']['num_dates_found']}")
    
    # Salvar visualização
    if result.get('visualization_base64'):
        vis = decode_base64_image(result['visualization_base64'])
        vis_path = "visualization.jpg"
        vis.save(vis_path)
        print(f"\n🎨 Visualização salva: {vis_path}")
    
    # Salvar crops
    crop_count = 0
    for i, ocr in enumerate(result['ocr_results'], 1):
        if ocr.get('crop_original_base64'):
            crop_original = decode_base64_image(ocr['crop_original_base64'])
            crop_original.save(f"full_crop_{i}_original.jpg")
            crop_count += 1
        
        if ocr.get('crop_processed_base64'):
            crop_processed = decode_base64_image(ocr['crop_processed_base64'])
            crop_processed.save(f"full_crop_{i}_processed.jpg")
            crop_count += 1
    
    print(f"🖼️  Crops salvos: {crop_count}")
    
    # Mostrar segmentação
    for i, detection in enumerate(result['detections'], 1):
        if detection.get('segmentation'):
            print(f"\n🎭 Segmentação {i}:")
            print(f"   Pontos: {len(detection['segmentation'])}")
            print(f"   Confiança: {detection['confidence']:.2%}")
    
    return result


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decodifica imagem base64."""
    # Remover prefixo data:image/...;base64,
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def save_segmentation_html(result: dict, output_path: str = "segmentation.html"):
    """
    Cria um HTML para visualizar a segmentação.
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Segmentação</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        .container {
            position: relative;
            display: inline-block;
        }
        svg {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        .info {
            margin-top: 20px;
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>🎭 Visualização de Segmentação</h1>
    <div class="container">
        <img src="IMAGE_PATH" id="mainImage" />
        <svg id="overlay"></svg>
    </div>
    <div class="info">
        <h2>📊 Informações</h2>
        <pre id="data"></pre>
    </div>
    
    <script>
        const data = DATA_JSON;
        
        // Configurar SVG
        const img = document.getElementById('mainImage');
        const svg = document.getElementById('overlay');
        
        img.onload = function() {
            svg.setAttribute('width', img.width);
            svg.setAttribute('height', img.height);
            
            // Desenhar segmentações
            data.detections.forEach((det, i) => {
                if (det.segmentation) {
                    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                    const points = det.segmentation.map(p => p.join(',')).join(' ');
                    polygon.setAttribute('points', points);
                    polygon.setAttribute('fill', 'rgba(0, 255, 0, 0.3)');
                    polygon.setAttribute('stroke', '#00ff00');
                    polygon.setAttribute('stroke-width', '2');
                    svg.appendChild(polygon);
                }
            });
        };
        
        // Mostrar dados
        document.getElementById('data').textContent = JSON.stringify(data, null, 2);
    </script>
</body>
</html>
    """
    
    html = html.replace('DATA_JSON', json.dumps(result))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n📄 HTML salvo: {output_path}")
    print(f"   Abra no navegador para ver a segmentação")


def main():
    """Executar todos os testes."""
    import sys
    
    if len(sys.argv) < 2:
        print("❌ Uso: python test_api_features.py <caminho_da_imagem>")
        print("   Exemplo: python test_api_features.py data/ocr_test/sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        sys.exit(1)
    
    print(f"\n🖼️  Imagem: {image_path}\n")
    
    # Teste 1: Apenas segmentação
    result1 = test_segmentation_only(image_path)
    
    # Teste 2: Com crops
    result2 = test_with_crop_images(image_path)
    
    # Teste 3: Pipeline completo
    result3 = test_full_pipeline(image_path)
    
    # Criar HTML de visualização
    save_segmentation_html(result3)
    
    print("\n" + "=" * 70)
    print("✅ TODOS OS TESTES CONCLUÍDOS")
    print("=" * 70)
    print("\nArquivos gerados:")
    print("  - crop_*_original.jpg")
    print("  - crop_*_processed.jpg")
    print("  - visualization.jpg")
    print("  - segmentation.html")
    print("\n💡 Abra segmentation.html no navegador para ver a segmentação")


if __name__ == "__main__":
    main()
