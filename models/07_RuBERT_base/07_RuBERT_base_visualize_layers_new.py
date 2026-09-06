#!/usr/bin/env python3
"""
t-SNE на 200 случайных примерах из тестовых данных
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# Загрузка модели
model_path = "./models"
print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Загружаем ваши предсказания (чтобы взять реальные названия и лейблы)
df = pd.read_csv("predictions_07.csv")  # У вас есть этот файл

# Берем случайные 100 Supply и 100 Work (итого 200)
supply_sample = df[df['true_label'] == 0].sample(n=100, random_state=42)
work_sample = df[df['true_label'] == 1].sample(n=100, random_state=42)
sample_df = pd.concat([supply_sample, work_sample])

texts = sample_df['ContractName'].tolist()  # или как называется колонка с текстом
labels = sample_df['true_label'].tolist()

print(f"Всего текстов: {len(texts)}")

# Получаем эмбеддинги ТОЛЬКО с последнего слоя (слой 12)
embeddings = []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs, output_hidden_states=True)
        cls_emb = outputs.hidden_states[-1][0, 0].cpu().numpy()
        embeddings.append(cls_emb)

embeddings = np.array(embeddings)

# t-SNE (может работать 1-2 минуты)
print("Запуск t-SNE для 200 точек...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
emb_2d = tsne.fit_transform(embeddings)

# Рисуем
plt.figure(figsize=(10, 8))
colors = ['#1f77b4' if l == 0 else '#d62728' for l in labels]
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=30, alpha=0.6)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.title('t-SNE визуализация эмбеддингов [CLS] (200 примеров)', fontsize=14)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', label='Supply'), Patch(facecolor='#d62728', label='Work')]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, alpha=0.3)
plt.savefig('tsne_many.png', dpi=300, bbox_inches='tight')
print("✅ СОХРАНЕНО: tsne_many.png")
plt.show()
