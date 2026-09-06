#!/usr/bin/env python3
"""
Массовый инференс для модели
"""

import json
import pickle
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("МАССОВЫЙ ИНФЕРЕНС")
print("=" * 60)

# 1. Загрузка данных
CONTRACTS_FILE = "/root/VKR/выгрузка_победителей/contracts_today_current.json"

with open(CONTRACTS_FILE, 'r', encoding='utf-8') as f:
    contracts = json.load(f)

print(f" Загружено {len(contracts)} контрактов")

# 2. Загрузка модели
model_dir = Path(__file__).parent

with open(model_dir / 'models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(model_dir / 'models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Модель и векторизатор загружены")

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

print(f"Обрабатывается {len(texts)} записей...")

# 4. Инференс
X = vectorizer.transform(texts)
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

# 5. Сохранение
model_name = model_dir.name[:2]
output_file = model_dir / f'predictions_{model_name}.csv'

for i, (p, prob) in enumerate(zip(preds, probs)):
    rows[i]['predicted_class'] = 'supply' if p == 1 else 'work'
    rows[i]['confidence'] = prob

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Результаты сохранены в {output_file}")
print(f"Всего предсказано: {len(df)}")
print(f"   supply: {len(df[df['predicted_class'] == 'supply'])}")
print(f"   work: {len(df[df['predicted_class'] == 'work'])}")
