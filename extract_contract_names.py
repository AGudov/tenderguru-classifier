#!/usr/bin/env python3
"""
Извлекает все ContractName из JSON-файлов в указанной папке
и сохраняет их в текстовый файл (по одному на строку) создавая датасет для прогрузки в Гигачат с целью создания заметки.
"""

import os
import json
import argparse
from pathlib import Path

def extract_names_from_json(file_path):
    """Читает JSON-файл и возвращает список ContractName (строк)"""
    names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Ошибка парсинга {file_path}, пропускаем")
            return names

        # Если data — список, перебираем элементы
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'ContractName' in item:
                    names.append(item['ContractName'])
        # Если data — словарь с ключом, например, "contracts"
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'ContractName' in item:
                            names.append(item['ContractName'])
                elif isinstance(value, dict) and 'ContractName' in value:
                    names.append(value['ContractName'])
        else:
            print(f"Неизвестный формат {file_path}")
    return names

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение ContractName из JSON-файлов с тендерами"
    )
    parser.add_argument(
        '--input', '-i',
        default='.',
        help='Путь к папке с JSON-файлами (по умолчанию текущая)'
    )
    parser.add_argument(
        '--output', '-o',
        default='contract_names_for_labeling.txt',
        help='Имя выходного текстового файла'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.json',
        help='Шаблон имён файлов (по умолчанию *.json)'
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Папка {input_dir} не существует")
        return

    # Собираем в
    files = list(input_dir.glob(args.pattern))
    if not files:
        print(f" Не найдено файлов по шаблону {args.pattern} в {input_dir}")
        return

    all_names = []
    for f in files:
        print(f"Обработка {f.name} ...")
        names = extract_names_from_json(f)
        all_names.extend(names)
        print(f"   Найдено {len(names)} названий")

    # Удаляем дубликаты 
    # all_names = list(dict.fromkeys(all_names))

    # Записываем в выходной файл
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as out:
        for name in all_names:
            out.write(name + '\n')

    print(f"Сохранено {len(all_names)} названий в {output_path}")

if __name__ == '__main__':
    main()
