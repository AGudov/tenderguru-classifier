#!/usr/bin/env python3
"""
Визуализация косинусного расстояния между эмбеддингами [CLS]
Показывает, что похожие тексты имеют близкие векторы
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Путь к модели 
model_path = "./models"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# ВАЖНО: включаем output_hidden_states, чтобы получить эмбеддинги
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    output_hidden_states=True
)
model.eval()
print("Модель загружена.")

# 6 показательных примеров (3 поставки, 3 работы)
texts = [
    # Поставки (Supply)
    "Поставка шпаклевки и краски для ремонта",
    "Поставка компьютерной техники для школы",
    "Поставка медицинских масок и перчаток",
    # Работы (Work)
    "Капитальный ремонт кровли школы",
    "Строительство автомобильной дороги",
    "Ремонт и отделка помещений поликлиники"
]

print("Тексты:")
for i, t in enumerate(texts):
    print(f"{i+1}: {t}")

# Функция получения эмбеддинга [CLS] с последнего слоя
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    # Берём эмбеддинг [CLS] из последнего скрытого слоя (индекс -1)
    cls_emb = outputs.hidden_states[-1][0, 0].cpu().numpy()
    return cls_emb

# Собираем эмбеддинги
embeddings = np.array([get_cls_embedding(t) for t in texts])

# Считаем косинусное сходство между всеми парами
sim_matrix = cosine_similarity(embeddings)

# Визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix,
            xticklabels=[f"T{i+1}" for i in range(len(texts))],
            yticklabels=[f"T{i+1}" for i in range(len(texts))],
            annot=True, fmt=".3f", cmap="viridis", square=True,
            cbar_kws={'label': 'Косинусное сходство'})

# Разделители между классами (после 3-го элемента)
plt.axhline(y=3, color='red', linestyle='--', linewidth=2)
plt.axvline(x=3, color='red', linestyle='--', linewidth=2)

plt.title('Косинусное сходство эмбеддингов [CLS]\nT1–T3: Supply (поставки), T4–T6: Work (работы)', fontsize=14)
plt.tight_layout()
plt.savefig('cosine_similarity.png', dpi=300)
print(" Визуализация сохранена: cosine_similarity.png")
plt.show()
