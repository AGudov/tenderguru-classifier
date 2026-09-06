#!/usr/bin/env python3
"""
Модель 1: Logistic Regression + TF-IDF
Бейзлайн для классификации тендеров
"""

import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print(" МОДЕЛЬ 1: LOGISTIC REGRESSION + TF-IDF")
print("=" * 60)

# =================== 1. ЗАГРУЗКА ДАННЫХ ===================
print("\n Загрузка датасета...")

with open('../tenders_labeled_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [1 if item['label'] == 'supply' else 0 for item in data]  # 1 = supply, 0 = work

print(f"   Загружено {len(texts)} записей")
print(f"   supply: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"   work: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

# =================== 2. РАЗБИЕНИЕ ===================
print("\n Разбиение на train/test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)

print(f"   Train: {len(X_train)}")
print(f"   Test: {len(X_test)}")

# =================== 3. TF-IDF ВЕКТОРИЗАЦИЯ ===================
print("\n Векторизация текстов (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words=None
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Размер матрицы: {X_train_tfidf.shape}")

# =================== 4. ОБУЧЕНИЕ ===================
print("\n Обучение Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)

print("   Модель обучена")

# =================== 5. ПРЕДСКАЗАНИЯ ===================
print("\n Оценка модели...")

y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("    Результаты на тестовой выборке:")
print(f"      Accuracy:  {accuracy:.4f}")
print(f"      Precision: {precision:.4f}")
print(f"      Recall:    {recall:.4f}")
print(f"      F1-Score:  {f1:.4f}")
print(f"      ROC-AUC:   {roc_auc:.4f}")

# =================== 6. CROSS-VALIDATION ===================
print("\n Cross-Validation (5-fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='f1')

print(f"   CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =================== 7. СОХРАНЕНИЕ ===================
print("\n Сохранение модели...")

os.makedirs('models', exist_ok=True)

# Сохраняем модель
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Сохраняем векторизатор
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("   Модель сохранена в models/")

# =================== 8. ОТЧЁТ ===================
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=['work', 'supply']))

# =================== 9. CONFUSION MATRIX ===================
print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Визуализация
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['work', 'supply'],
            yticklabels=['work', 'supply'])
plt.title('Confusion Matrix: LogReg + TF-IDF')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150)
print("   Confusion Matrix сохранён: confusion_matrix.png")

# =================== 10. ИТОГ ===================
print("\n" + "=" * 60)
print(" ИТОГИ МОДЕЛИ 1")
print("=" * 60)
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print("=" * 60)

# Сохраняем метрики
with open('results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
    f.write(f"CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

print("    Результаты сохранены в results.txt")
