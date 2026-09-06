#!/usr/bin/env python3
"""
Модель 6: CatBoost + RuBERT (дообученный на тендерах)
Эмбеддинги от дообученного RuBERT + градиентный бустинг (CatBoost)
"""

import json
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📊 МОДЕЛЬ 6: CATBOOST + RuBERT (ДООБУЧЕННЫЙ)")
print("=" * 60)

# =================== 1. ЗАГРУЗКА ДАННЫХ ===================
print("\n📂 Загрузка датасета...")

with open('../tenders_labeled_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [1 if item['label'] == 'supply' else 0 for item in data]

print(f"   ✅ Загружено {len(texts)} записей")
print(f"   ✅ supply: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"   ✅ work: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

# =================== 2. РАЗБИЕНИЕ ===================
print("\n📊 Разбиение на train/test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"   ✅ Train: {len(X_train)}")
print(f"   ✅ Test: {len(X_test)}")

# =================== 3. ЗАГРУЗКА ДООБУЧЕННОЙ МОДЕЛИ ===================
print("\n🧠 Загрузка дообученного RuBERT...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   ✅ Используется: {device}")

# ПУТЬ К ДООБУЧЕННОЙ МОДЕЛИ (укажи правильный путь!)
finetuned_model_path = "/root/VKR/модель_классификации/rubert_finetuned"

try:
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModel.from_pretrained(finetuned_model_path).to(device)
    print("   ✅ Загружена ДООБУЧЕННАЯ модель")
except:
    print("   ⚠️ Дообученная модель не найдена, используется базовая RuBERT")
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(device)

# =================== 4. ФУНКЦИЯ ПОЛУЧЕНИЯ ЭМБЕДДИНГОВ ===================
def get_embedding(text):
    """Получает эмбеддинг текста через RuBERT"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embedding

# =================== 5. СОЗДАНИЕ ЭМБЕДДИНГОВ ===================
print("\n🔄 Создание эмбеддингов для train (9218 текстов)...")
start_time = time.time()

X_train_emb = []
for i, text in enumerate(X_train):
    if i % 500 == 0:
        print(f"   Обработано {i}/{len(X_train)}")
    X_train_emb.append(get_embedding(text))

X_train_emb = np.vstack(X_train_emb)

print(f"   ✅ Размер эмбеддингов: {X_train_emb.shape}")

print("\n🔄 Создание эмбеддингов для test (2305 текстов)...")
X_test_emb = []
for i, text in enumerate(X_test):
    if i % 500 == 0:
        print(f"   Обработано {i}/{len(X_test)}")
    X_test_emb.append(get_embedding(text))

X_test_emb = np.vstack(X_test_emb)

print(f"   ✅ Размер эмбеддингов: {X_test_emb.shape}")

embed_time = time.time() - start_time
print(f"   ⏱️ Время создания эмбеддингов: {embed_time:.2f} сек")

# =================== 6. ОБУЧЕНИЕ CATBOOST ===================
print("\n🤖 Обучение CatBoost на эмбеддингах...")

start_time = time.time()

model_cb = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=50,
    early_stopping_rounds=50
)

model_cb.fit(X_train_emb, y_train)

train_time = time.time() - start_time
print(f"   ✅ Модель обучена за {train_time:.2f} сек")

# =================== 7. ПРЕДСКАЗАНИЯ ===================
print("\n📈 Оценка модели...")

y_pred = model_cb.predict(X_test_emb)
y_proba = model_cb.predict_proba(X_test_emb)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("   📊 Результаты на тестовой выборке:")
print(f"      Accuracy:  {accuracy:.4f}")
print(f"      Precision: {precision:.4f}")
print(f"      Recall:    {recall:.4f}")
print(f"      F1-Score:  {f1:.4f}")
print(f"      ROC-AUC:   {roc_auc:.4f}")

# =================== 8. CROSS-VALIDATION ===================
print("\n🔄 Cross-Validation (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_cb, X_train_emb, y_train, cv=cv, scoring='f1')

print(f"   ✅ CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =================== 9. СОХРАНЕНИЕ ===================
print("\n💾 Сохранение модели...")

os.makedirs('models', exist_ok=True)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model_cb, f)

print("   ✅ Модель сохранена в models/")

# =================== 10. CONFUSION MATRIX ===================
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['work', 'supply'],
            yticklabels=['work', 'supply'])
plt.title('Confusion Matrix: CatBoost + RuBERT (finetuned)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150)
print("   ✅ Confusion Matrix сохранён: confusion_matrix.png")

# =================== 11. ИТОГ ===================
print("\n" + "=" * 60)
print("📊 ИТОГИ МОДЕЛИ 6")
print("=" * 60)
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Время эмбеддингов: {embed_time:.2f} сек")
print(f"   Время обучения:    {train_time:.2f} сек")
print("=" * 60)

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
    f.write(f"CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    f.write(f"Время эмбеддингов: {embed_time:.2f} сек\n")
    f.write(f"Время обучения:    {train_time:.2f} сек\n")

print("   ✅ Результаты сохранены в results.txt")
