#!/usr/bin/env python3
"""
Инференс для полной нейросети RuBERT (base) с сохранением результата в CSV
Поддерживает режим --test (только первые N записей)
"""

import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# =================== ПАРСИНГ АРГУМЕНТОВ ===================
test_mode = False
limit = None

for arg in sys.argv:
    if arg == '--test':
        test_mode = True
    elif arg.startswith('--limit='):
        try:
            limit = int(arg.split('=')[1])
        except:
            pass

print("=" * 60)
print(" ИНФЕРЕНС: RuBERT (полная нейросеть, BASE)")
print("=" * 60)
if test_mode:
    print(" РЕЖИМ ТЕСТА: первые {} записей".format(limit if limit else 50))

# 1. Загрузка данных
CONTRACTS_FILE = "/root/VKR/выгрузка_победителей/contracts_today_current.json"

with open(CONTRACTS_FILE, 'r', encoding='utf-8') as f:
    contracts = json.load(f)

if test_mode:
    if limit is None:
        limit = 50
    contracts = contracts[:limit]
    print(f" Тестовый режим: {len(contracts)} записей")
else:
    print(f" Загружено {len(contracts)} контрактов")

# 2. Загрузка модели RuBERT
model_dir = Path(__file__).parent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Используется: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_dir / 'models')
model = AutoModelForSequenceClassification.from_pretrained(model_dir / 'models')
model = model.to(device)
model.eval()

print(" RuBERT (base) загружен")

# 3. Подготовка данных
texts = []
rows = []

for item in contracts:
    name = item.get('ContractName', '').strip()
    if name:
        texts.append(name)
        rows.append({
            'id': item.get('ID', ''),
            'contract_name': name,
            'customer': item.get('Customer', ''),
            'region': item.get('Region', ''),
        })

print(f" Обрабатывается {len(texts)} записей...")

# 4. Инференс с прогресс-баром
preds = []
probs = []

for text in tqdm(texts, desc="Инференс RuBERT", unit="текст"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0][pred]
    
    preds.append(pred)
    probs.append(prob)

# 5. Сохранение
model_name = model_dir.name[:2]
output_file = model_dir / f'predictions_{model_name}.csv'

for i, (p, prob) in enumerate(zip(preds, probs)):
    rows[i]['predicted_class'] = 'supply' if p == 1 else 'work'
    rows[i]['confidence'] = prob

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f" Результаты сохранены в {output_file}")
print(f"  Всего предсказано: {len(df)}")

if not test_mode:
    print(f"   supply: {len(df[df['predicted_class'] == 'supply'])}")
    print(f"   work: {len(df[df['predicted_class'] == 'work'])}")
else:
    print("\n  ПРИМЕРЫ:")
    for _, row in df.head(10).iterrows():
        print(f"   {row['predicted_class']:6} | {row['confidence']:.2%} | {row['contract_name'][:50]}...")
