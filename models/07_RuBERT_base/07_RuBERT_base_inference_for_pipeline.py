#!/usr/bin/env python3
"""
Инференс для конвейера v3:
- Берёт самый свежий файл из 01_Победители
- Классифицирует все контракты
- Сохраняет с тем же именем, но с суффиксами
- Копирует поставки в папку поставки_общая
"""

import json
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import sys
import re

# =================== ПУТИ ===================
CONTRACTS_DIR = "/root/VKR/Конвейер/01_Победители"
OUTPUT_DIR = "/root/VKR/Конвейер/02_Классификация/классификация"
SUPPLY_ALL_DIR = "/root/VKR/Конвейер/02_Классификация/поставки_общая"
LOG_DIR = "/root/VKR/Конвейер/12_Логи"
MODEL_DIR = "/root/VKR/work_supply_split/модели_для_вкр_сравнение/07_RuBERT_base/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUPPLY_ALL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "02_inference.log")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')

# =================== 1. НАХОДИМ САМЫЙ СВЕЖИЙ ФАЙЛ ===================
def get_latest_file(directory):
    pattern = os.path.join(directory, "contracts_today_*.json")
    files = glob.glob(pattern)
    current_file = os.path.join(directory, "contracts_today_current.json")
    
    if os.path.exists(current_file):
        files.append(current_file)
    
    if not files:
        return None
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# =================== 2. ЗАГРУЗКА МОДЕЛИ ===================
log("=" * 60)
log(" ЗАПУСК ИНФЕРЕНСА v3")
log("=" * 60)

log(" Загрузка модели RuBERT...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f" Используется: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model = model.to(device)
    model.eval()
    log(" Модель загружена")
except Exception as e:
    log(f"  Ошибка загрузки модели: {e}")
    sys.exit(1)

# =================== 3. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ===================
def predict_class(text):
    if not text or not text.strip():
        return "unknown", 0.0
    
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
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    class_name = "supply" if pred == 1 else "work"
    confidence = proba[pred]
    
    return class_name, confidence

# =================== 4. ОБРАБОТКА ===================
contracts_file = get_latest_file(CONTRACTS_DIR)
if not contracts_file:
    log(" Нет файлов контрактов в папке 01_Победители")
    sys.exit(1)

log(f" Входной файл: {contracts_file}")

# Извлекаем базовое имя файла
base_name = os.path.basename(contracts_file)
# contracts_today_20260815_153612.json - tender_today_20260815_153612
base_name_no_ext = base_name.replace('.json', '').replace('contracts_', 'tender_')

log(f" Базовое имя: {base_name_no_ext}")

with open(contracts_file, 'r', encoding='utf-8') as f:
    contracts = json.load(f)

log(f" Загружено {len(contracts)} контрактов")

log(" Классификация...")
classified = []
supply_count = 0
work_count = 0

for i, contract in enumerate(contracts, 1):
    if i % 100 == 0:
        log(f"   Обработано {i}/{len(contracts)}")
    
    contract_name = contract.get('ContractName', '')
    class_name, confidence = predict_class(contract_name)
    
    contract['predicted_class'] = class_name
    contract['confidence'] = float(confidence)
    
    classified.append(contract)
    
    if class_name == 'supply':
        supply_count += 1
    else:
        work_count += 1

log(f"    Обработано {len(classified)} контрактов")
log(f"    supply: {supply_count}")
log(f"    work: {work_count}")

# =================== 5. СОХРАНЯЕМ ===================

# 5.1 Все контракты
output_file = os.path.join(OUTPUT_DIR, f"{base_name_no_ext}_classified.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(classified, f, ensure_ascii=False, indent=2)
log(f" Все контракты: {output_file}")

# 5.2 Только поставки
supply_only = [c for c in classified if c.get('predicted_class') == 'supply']
supply_file = os.path.join(OUTPUT_DIR, f"{base_name_no_ext}_supply.json")
with open(supply_file, 'w', encoding='utf-8') as f:
    json.dump(supply_only, f, ensure_ascii=False, indent=2)
log(f" Поставки ({len(supply_only)}): {supply_file}")

# 5.3 Только работы
work_only = [c for c in classified if c.get('predicted_class') == 'work']
work_file = os.path.join(OUTPUT_DIR, f"{base_name_no_ext}_work.json")
with open(work_file, 'w', encoding='utf-8') as f:
    json.dump(work_only, f, ensure_ascii=False, indent=2)
log(f" Работы ({len(work_only)}): {work_file}")

# 5.4 Копируем поставки в папку поставки_общая
supply_all_file = os.path.join(SUPPLY_ALL_DIR, f"{base_name_no_ext}_supply_only.json")
with open(supply_all_file, 'w', encoding='utf-8') as f:
    json.dump(supply_only, f, ensure_ascii=False, indent=2)
log(f" Копия поставок в общую папку: {supply_all_file}")

log("=" * 60)
log(" ГОТОВО!")
log("=" * 60)
