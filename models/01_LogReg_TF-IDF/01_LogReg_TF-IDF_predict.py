#!/usr/bin/env python3
"""
Инференс для модели 1: Logistic Regression + TF-IDF
Вводишь текст → модель определяет: supply или work
"""

import pickle
import sys
import os

# =================== ЗАГРУЗКА МОДЕЛИ ===================
print("=" * 60)
print("ЗАГРУЗКА МОДЕЛИ LogReg + TF-IDF")
print("=" * 60)

try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Модель загружена")

    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Векторизатор загружен")
except FileNotFoundError:
    print("Ошибка: модель не найдена. Сначала обучи модель (python3 train.py)")
    sys.exit(1)

print("=" * 60)

# =================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ===================
def predict(text):
    """
    Предсказывает класс для одного текста
    """
    # 1. Превращаем текст в вектор (TF-IDF)
    text_vector = vectorizer.transform([text])
    
    # 2. Предсказываем класс
    class_id = model.predict(text_vector)[0]
    class_name = "supply" if class_id == 1 else "work"
    
    # 3. Предсказываем вероятность
    proba = model.predict_proba(text_vector)[0]
    proba_supply = proba[1] if class_id == 1 else proba[0]
    proba_work = proba[0] if class_id == 0 else proba[1]
    
    return {
        'class': class_name,
        'class_id': class_id,
        'confidence': proba_supply if class_id == 1 else proba_work,
        'proba_work': proba[0],
        'proba_supply': proba[1]
    }

# =================== ИНТЕРАКТИВНЫЙ РЕЖИМ ===================
print("\nВВЕДИТЕ НАЗВАНИЕ ТЕНДЕРА")
print("   (введите 'exit' для выхода)")
print("-" * 60)

while True:
    text = input("\nВведите название: ").strip()
    
    if text.lower() == 'exit':
        print("\nДо свидания!")
        break
    
    if not text:
        print("Пожалуйста, введите текст")
        continue
    
    # Предсказание
    result = predict(text)
    
    # Вывод
    print("\n" + "=" * 60)
    print(f"Результат: {result['class'].upper()}")
    print("=" * 60)
    print(f"   Класс: {result['class']}")
    print(f"   Уверенность: {result['confidence']:.2%}")
    print(f"   Вероятность work: {result['proba_work']:.2%}")
    print(f"   Вероятность supply: {result['proba_supply']:.2%}")
    print("=" * 60)
