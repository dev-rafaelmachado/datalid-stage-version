import json

data = json.load(open('outputs/pipeline_evaluation/20251105_014608/evaluation_results.json'))
errors = [r for r in data['results'] if r['exact_match'] == 0.0]

print(f"Total de erros: {len(errors)}/50")
print("\n" + "="*100)
print("ANÁLISE DOS ERROS:")
print("="*100 + "\n")

for i, r in enumerate(errors[:20], 1):
    gt = r['ground_truth']
    pred = r['predicted_date']
    ocr = r['ocr_texts'][0] if r['ocr_texts'] else "EMPTY"
    
    print(f"{i}. GT: {gt} | PRED: {pred}")
    print(f"   OCR: '{ocr}'")
    print(f"   Imagem: {r['image_number']}")
    print()
