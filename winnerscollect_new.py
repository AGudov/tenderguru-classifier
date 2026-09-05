#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сбор контрактов за сегодня (today=1) с API tenderguru.ru
Сохраняет промежуточные результаты после каждой страницы.
"""

import json
import time
import requests
import urllib3
from datetime import datetime
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Жёстко заданный URL
URL = "https://www.tenderguru.ru/api2.3/export/contracts?today=1&dtype=json&api_code=***********"

BASE_DIR = os.path.expanduser("~/VKR")
OUTPUT_DIR = os.path.join(BASE_DIR, "Конвейер/01_Победители")
LOG_FILE = os.path.join(BASE_DIR, "logs", "collect.log")

DELAY = 1.5  # задержка между страницами

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} - {msg}\n")
    print(f"[{timestamp}] {msg}")

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_current(contracts):
    """Сохраняет текущий список как 'contracts_today_current.json' (перезаписывает)."""
    filepath = os.path.join(OUTPUT_DIR, "contracts_today_current.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(contracts, f, ensure_ascii=False, indent=2)
    log(f"Обновлён текущий файл: {len(contracts)} записей")

def save_final(contracts):
    if not contracts:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"contracts_today_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(contracts, f, ensure_ascii=False, indent=2)
    log(f" Финальный файл сохранён: {filename} ({len(contracts)} записей)")

def fetch_page(page):
    """Загружает одну страницу по фиксированному URL с добавлением page."""
    url = f"{URL}&page={page}"
    try:
        log(f"Загрузка страницы {page}...")
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if data.get('status') == 'error':
                log(f" Ошибка API: {data.get('message', '')}")
                return None
            if data.get('total', 0) == 0:
                log(f" Страница {page} пуста (total=0)")
                return None
            contracts = data.get('items', [])
            if not contracts:
                contracts = data.get('contracts', [])
            return contracts
        else:
            log(f"Неожиданный тип данных: {type(data)}")
            return None
    except Exception as e:
        log(f" Ошибка загрузки страницы {page}: {e}")
        return None

def main():
    log("=" * 60)
    log("ЗАГРУЗКА КОНТРАКТОВ ЗА СЕГОДНЯ (today=1)")
    log("=" * 60)
    ensure_output_dir()

    all_contracts = []
    page = 1
    while True:
        contracts = fetch_page(page)
        if contracts is None or not contracts:
            log(f" Страница {page} пуста, завершаем")
            break
        all_contracts.extend(contracts)
        log(f"Страница {page}: +{len(contracts)} (всего: {len(all_contracts)})")
        # Сохраняем промежуточный результат после каждой страницы
        save_current(all_contracts)
        page += 1
        time.sleep(DELAY)

    if all_contracts:
        save_final(all_contracts)
        log(f"Всего сохранено контрактов: {len(all_contracts)}")
    else:
        log(" Не удалось загрузить контракты")

    log("=" * 60)
    log(" ЗАВЕРШЕНИЕ")
    log("=" * 60)

if __name__ == "__main__":
    main()
