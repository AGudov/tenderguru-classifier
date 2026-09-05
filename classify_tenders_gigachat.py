#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Массовая классификация тендеров через GigaChat-Max
"""

import requests
import json
import uuid
import time
from tqdm import tqdm


AUTH_KEY = '*****'
RQUID = '****'
MODEL = "GigaChat-Max"

# 2. ПОЛУЧЕНИЕ ТОКЕНА

def get_token():
    resp = requests.post(
        'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': RQUID,
            'Authorization': f'Basic {AUTH_KEY}'
        },
        data={'scope': 'GIGACHAT_API_PERS'},
        verify=False
    )
    return resp.json()['access_token']

# 3. ПРОМПТ ДЛЯ КЛАССИФИКАЦИИ НАИМЕНОВАНИЯ ТЕНДЕРА

SYSTEM_PROMPT = """Ты — классификатор тендеров. Определи, является ли тендер поставкой товара (supply) или выполнением работ/услуг (work).

## КЛАССЫ:
- **supply (поставка)**: передача готового товара, материала, оборудования, продукции.
- **work (работы)**: процесс, действие, услуга, ремонт, монтаж, строительство.

## ПРИМЕРЫ:
### supply:
- «Молоко питьевое», «Ноутбук», «Кирпич», «Лекарственные препараты», «Бензин АИ-92», «Шприцы», «Приобретение жилого помещения», «Поставка реагентов»

### work:
- «Капитальный ремонт кровли», «Монтаж сигнализации», «Благоустройство территории», «Услуги по перевозке пассажиров», «Уборка территории», «Техническое обслуживание», «Проектирование», «Обследование зданий»

## КЛЮЧЕВЫЕ СЛОВА ДЛЯ work (если есть → work):
ремонт, капитальный ремонт, текущий ремонт, аварийный ремонт,
монтаж, демонтаж, установка, замена, переустройство,
строительство, реконструкция, реставрация, капитальное строительство,
проектирование, изыскания, обследование, диагностика,
покраска, отделка, пусконаладка, наладка, калибровка, поверка,
перевозка, вывоз, утилизация, обработка, испытание, проверка,
очистка, стирка, химчистка, дезинфекция, дератизация, дезинсекция,
обслуживание, техническое обслуживание, заправка (картриджей),
обучение, консультирование, охрана, питание,
благоустройство, озеленение, устройство (покрытий, оснований),
разработка (документации, ПО), восстановление (зданий, сооружений),
приспособление (для нужд маломобильных групп).

## КЛЮЧЕВЫЕ СЛОВА ДЛЯ supply (если есть → supply):
поставка, приобретение, продажа, передача,
товар, продукция, изделие, материал, сырьё, полуфабрикат,
оборудование, техника, автомобиль, запчасти, комплектующие,
инструмент, станок, прибор, аппарат, устройство,
мебель, оргтехника, компьютеры, комплектующие, канцтовары,
одежда, обувь, спецодежда, СИЗ,
медикаменты, лекарства, вакцины, реагенты, тест-системы,
продовольствие, продукты питания, напитки,
стройматериалы, ГСМ, расходные материалы,
штука, метр, килограмм, тонна, литр, упаковка, набор, комплект.

## РЕШАЮЩИЕ ПРАВИЛА:
1. **Если есть слово «ремонт», «монтаж», «строительство», «благоустройство», «капитальный» → work.**
2. **Если есть слово «поставка», «приобретение» + название товара → supply.**
3. **Если есть процесс (глагол: ремонтировать, строить, устанавливать, перевозить) → work.**
4. **Если название — это название предмета (ноутбук, труба, молоко) → supply.**
5. **«Товарный знак: Отсутствует» — игнорируй, смотри по смыслу.**
6. **Смешанные случаи («капитальный ремонт с поставкой материалов» или "поставка и ремонт") → work (главное — процесс).**
7. **Если не уверен — отдай предпочтение supply (консервативный подход).**

Ответь только одним словом: supply или work. Без кавычек, пояснений, дополнительного текста."""

def classify_one(text, max_retries=3):
    """Классифицирует один тендер с повторными попытками при ошибках."""
    for attempt in range(max_retries):
        try:
            token = get_token()
            resp = requests.post(
                'https://gigachat.devices.sberbank.ru/api/v1/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {token}'
                },
                json={
                    'model': MODEL,
                    'messages': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': text}
                    ],
                    'temperature': 0.1,
                    'max_tokens': 10
                },
                verify=False,
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()['choices'][0]['message']['content'].strip().lower()
            if result in ['supply', 'work']:
                return result
            else:
                print(f"⚠️ Неожиданный ответ: '{result}', повторная попытка...")
                time.sleep(1)
        except Exception as e:
            print(f"⚠️ Ошибка (попытка {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return 'unknown'
    return 'unknown'

# 4. ОСНОВНАЯ ФУНКЦИЯ

def main():
    print("Чтение tenders_for_gigachat.txt...")
    with open('tenders_for_gigachat.txt', 'r', encoding='utf-8') as f:
        tenders = [line.strip() for line in f if line.strip()]

    print(f" Найдено {len(tenders)} тендеров")
    print(f" Модель: {MODEL}")
    print(" Начинаем классификацию...\n")

    results = []
    for idx, text in enumerate(tqdm(tenders, desc="Классификация"), 1):
        label = classify_one(text)
        results.append({"text": text, "label": label})

        # Задержка между запросами
        time.sleep(0.5)

        # Промежуточное сохранение каждые 50 запросов
        if idx % 50 == 0:
            with open('tenders_labeled_partial.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Промежуточный результат сохранён ({idx}/{len(tenders)})")

    # Финальное сохранение
    with open('tenders_labeled_by_gigachat.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Статистика
    labels = [r['label'] for r in results if r['label'] in ['supply', 'work']]
    unknown = len([r for r in results if r['label'] not in ['supply', 'work']])
    
    print("\n" + "="*50)
    print("СТАТИСТИКА КЛАССИФИКАЦИИ")
    print("="*50)
    print(f"Всего обработано: {len(results)}")
    if labels:
        print(f"  supply: {labels.count('supply')} ({labels.count('supply')/len(labels)*100:.1f}%)")
        print(f"  work: {labels.count('work')} ({labels.count('work')/len(labels)*100:.1f}%)")
    else:
        print("  supply: 0 (0.0%)")
        print("  work: 0 (0.0%)")
    print(f"  unknown/error: {unknown}")
    print(f"\n Результат сохранён в tenders_labeled_by_gigachat.json")

if __name__ == "__main__":
    main()
