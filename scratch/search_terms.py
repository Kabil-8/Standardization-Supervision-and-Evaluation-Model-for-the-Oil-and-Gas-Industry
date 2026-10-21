with open('scratch/full_revised_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

terms = ['0.4743', '0.7195', 'hist_gradient_boosting', '3545', '2267', '31065', 'run_manifest', 'Kaggle', '22,957', '22957']
for term in terms:
    print(f'Term "{term}" in doc: {term in text}')
