#!/usr/bin/env python3
"""
ФИНАЛЬНАЯ визуализация: 3 слоя RuBERT (0, 6, 12) на 60 тендерах
Путь к модели исправлен на ./models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ПРАВИЛЬНЫЙ ПУТЬ К МОДЕЛИ (лежит в подпапке models)
model_path = "./models"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Модель загружена.")

# ГЕНЕРИРУЕМ 60 РЕАЛИСТИЧНЫХ НАЗВАНИЙ (30 Supply, 30 Work)
supply_texts = [
    "Поставка компьютерной техники", "Поставка офисной мебели", "Поставка канцелярских товаров",
    "Поставка медицинских масок", "Поставка стройматериалов", "Поставка продуктов питания",
    "Поставка оргтехники", "Поставка лекарственных препаратов", "Поставка школьных учебников",
    "Поставка автомобильных шин", "Поставка холодильного оборудования", "Поставка бумаги",
    "Поставка светильников", "Поставка спортивного инвентаря", "Поставка посуды",
    "Поставка компьютеров", "Поставка мебели для школы", "Поставка оконных блоков",
    "Поставка трубопровода", "Поставка кабельной продукции", "Поставка насосного оборудования",
    "Поставка электроинструмента", "Поставка расходных материалов", "Поставка упаковки",
    "Поставка мешков для мусора", "Поставка чистящих средств", "Поставка перчаток",
    "Поставка масок медицинских", "Поставка стульев офисных", "Поставка столов"
]

work_texts = [
    "Капитальный ремонт школы", "Строительство автомобильной дороги", "Ремонт кровли больницы",
    "Благоустройство городского парка", "Отделка помещений поликлиники", "Монтаж вентиляции",
    "Установка пожарной сигнализации", "Реконструкция моста", "Проектирование здания",
    "Капитальный ремонт фасада", "Строительство жилого дома", "Ремонт инженерных сетей",
    "Благоустройство дворовой территории", "Монтаж лифтового оборудования", "Установка кондиционеров",
    "Ремонт дорожного покрытия", "Строительство детского сада", "Отделка фасада",
    "Ремонт электропроводки", "Монтаж систем отопления", "Установка видеонаблюдения",
    "Ремонт водопровода", "Строительство спортивной площадки", "Укладка асфальта",
    "Ремонт подъездов", "Благоустройство набережной", "Монтаж слаботочных систем",
    "Строительство трансформаторной подстанции", "Ремонт кровли", "Отделка стен"
]

all_texts = supply_texts + work_texts
labels = [0] * len(supply_texts) + [1] * len(work_texts)
print(f"Всего текстов: {len(all_texts)}")

def get_embeddings_from_layer(texts, layer_idx):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64, padding=True)
            outputs = model(**inputs, output_hidden_states=True)
            layer_emb = outputs.hidden_states[layer_idx][0, 0].cpu().numpy()
            embeddings.append(layer_emb)
    return np.array(embeddings)

layers = [0, 6, 12]
layer_names = ['Слой 0 (входные эмбеддинги слов)', 'Слой 6 (середина трансформера)', 'Слой 12 (выход)']
all_emb_layers = [get_embeddings_from_layer(all_texts, l) for l in layers]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#1f77b4' if l == 0 else '#d62728' for l in labels]

for i, (emb, name) in enumerate(zip(all_emb_layers, layer_names)):
    tsne = TSNE(n_components=2, random_state=42, perplexity=8, max_iter=1000)
    emb_2d = tsne.fit_transform(emb)
    
    axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[i].set_title(f'{name}', fontsize=14)
    axes[i].set_xlabel('t-SNE Component 1', fontsize=11)
    axes[i].set_ylabel('t-SNE Component 2', fontsize=11)
    axes[i].grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Supply (поставки)'),
                       Patch(facecolor='#d62728', label='Work (работы)')]
    axes[i].legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.suptitle('Изменение векторных представлений тендеров по слоям RuBERT (t-SNE)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('tsne_layers_final.png', dpi=300, bbox_inches='tight')
print("✅ ВИЗУАЛИЗАЦИЯ СОХРАНЕНА: tsne_layers_final.png")
plt.show()
