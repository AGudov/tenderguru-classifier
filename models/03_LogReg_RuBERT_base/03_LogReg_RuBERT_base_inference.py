#!/usr/bin/env python3
"""
Инференс для RuBERT + ML с сохранением результата в CSV
Поддерживает режим --test (только первые N записей)
"""

import json
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
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
print(" ИНФЕРЕНС: RuBERT + ML")
print("=" * 60)
if test_mode:
    print("РЕЖИМ ТЕСТА: первые {} записей".format(limit if limit else 50))

# 1. Загрузка данных
CONTRACTS_FILE = "/root/VKR/выгрузка_победителей/contracts_today_current.json"

with open(CONTRACTS_FILE, 'r', encoding='utf-8') as f:
    contracts = json.load(f)

# Ограничение для теста
if test_mode:
    if limit is None:
        limit = 50
    contracts = contracts[:limit]
    print(f" Тестовый режим: {len(contracts)} записей")
else:
    print(f"Загружено {len(contracts)} контрактов")

# 2. Загрузка ML-модели
model_dir = Path(__file__).parent

with open(model_dir / 'models/model.pkl', 'rb') as f:
    ml_model = pickle.load(f)

# 3. Загрузка RuBERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Используется: {device}")

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(device)
bert_model.eval()

print(" RuBERT загружен")

# 4. Функция получения эмбеддинга
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

# 5. Подготовка данных
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

# 6. Инференс с прогресс-баром
embeddings = []
for text in tqdm(texts, desc="Создание эмбеддингов", unit="текст"):
    embeddings.append(get_embedding(text))

X = np.vstack(embeddings)
preds = ml_model.predict(X)
probs = ml_model.predict_proba(X)[:, 1]

# 7. Сохранение
model_name = model_dir.name[:2]
output_file = model_dir / f'predictions_{model_name}.csv'

for i, (p, prob) in enumerate(zip(preds, probs)):
    rows[i]['predicted_class'] = 'supply' if p == 1 else 'work'
    rows[i]['confidence'] = prob

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f" Результаты сохранены в {output_file}")
print(f" Всего предсказано: {len(df)}")

if not test_mode:
    print(f"   supply: {len(df[df['predicted_class'] == 'supply'])}")
    print(f"   work: {len(df[df['predicted_class'] == 'work'])}")
else:
    print("\n ПРИМЕРЫ:")
    for _, row in df.head(10).iterrows():
        print(f"   {row['predicted_class']:6} | {row['confidence']:.2%} | {row['contract_name'][:50]}...")
