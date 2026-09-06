#!/usr/bin/env python3
"""
Скрипт для демонстрации внутреннего устройства модели RuBERT.
Показывает токенизацию, эмбеддинги, логиты (без внимания, чтобы избежать ошибок).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Загружаем локальную модель с attn_implementation="eager"
model_path = "./models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    attn_implementation="eager"  # включаем вывод внимания
)
model.eval()

# Примеры текстов
texts = [
    "Капитальный ремонт школы",
    "Поставка шпаклевки и проведение ремонтных  работ в школе"
]

for text in texts:
    print("\n" + "=" * 60)
    print(f" Текст: {text}")
    print("=" * 60)

    # 1. Токенизация
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    print("\n Токены и их ID:")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    for i, (tid, token) in enumerate(zip(inputs['input_ids'][0], tokens)):
        print(f"   {i:2d}: {token:12} → {tid.item()}")

    # 2. Прогон через модель с выводом скрытых состояний
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False  # отключаем, чтобы избежать ошибок
        )

    # 3. Вектор [CLS] (эмбеддинг всего текста)
    cls_embedding = outputs.hidden_states[-1][0, 0, :10]  # первые 10 чисел
    print("\n Вектор [CLS] (первые 10 чисел из 768):")
    print(f"   {cls_embedding.tolist()}")

    # 4. Логиты и вероятности
    logits = outputs.logits
    proba = torch.softmax(logits, dim=1)
    class_id = torch.argmax(logits, dim=1).item()
    class_name = "supply" if class_id == 1 else "work"
    confidence = proba[0, class_id].item()

    print("\n Выход модели:")
    print(f"   Логиты: {logits.tolist()}")
    print(f"   Вероятности: {proba.tolist()}")
    print(f"   Класс: {class_name} (уверенность: {confidence:.2%})")
