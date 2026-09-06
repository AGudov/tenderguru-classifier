#!/usr/bin/env python3
"""
Визуализация весов внимания (последний слой, усреднённый по головам)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./models"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
model.eval()
print("Модель загружена.")

# ПРИМЕР
text = "Поставка шпаклевки с дальнейшим ремонтом учебного заведения"

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"Текст: {text}")
print(f"Токены: {tokens}")

with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

last_layer_attention = attentions[-1]
avg_attention = last_layer_attention.mean(dim=1)
attn_matrix = avg_attention[0].cpu().numpy()

max_len = min(attn_matrix.shape[0], 25)
attn_matrix = attn_matrix[:max_len, :max_len]
tokens_subset = tokens[:max_len]

plt.figure(figsize=(14, 12))
sns.heatmap(attn_matrix,
            xticklabels=tokens_subset,
            yticklabels=tokens_subset,
            cmap='Blues',
            square=True,
            annot=False,
            cbar_kws={'label': 'Вес внимания'})
plt.title(f'Внимание последнего слоя (усреднённое по головам)\nТекст: "{text}"', fontsize=14)
plt.xlabel('Токены (ключ)')
plt.ylabel('Токены (запрос)')
plt.tight_layout()
plt.savefig('attention_heatmap.png', dpi=300)
print(" Визуализация внимания сохранена: attention_heatmap.png")
plt.show()
