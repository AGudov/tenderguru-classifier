#!/usr/bin/env python3
"""
Модель 7: RuBERT (полная нейросеть, без дообучения)
Классификация через дообучение последнего слоя RuBERT
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" МОДЕЛЬ 7: RuBERT (ПОЛНАЯ НЕЙРОСЕТЬ, БЕЗ ДООБУЧЕНИЯ)")
print("=" * 60)

# =================== 1. ЗАГРУЗКА ДАННЫХ ===================
print("\n Загрузка датасета...")

with open('../tenders_labeled_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [1 if item['label'] == 'supply' else 0 for item in data]  # 1 = supply, 0 = work

print(f"    Загружено {len(texts)} записей")
print(f"   supply: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"    work: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

# =================== 2. РАЗБИЕНИЕ ===================
print("\n Разбиение на train/val/test (70/15/15)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"    Train: {len(X_train)}")
print(f"    Val: {len(X_val)}")
print(f"    Test: {len(X_test)}")

# =================== 3. ПОДГОТОВКА ДАТАСЕТА ===================
print("\n Подготовка датасета...")

# Создаем словари
train_data = {'text': X_train, 'label': y_train}
val_data = {'text': X_val, 'label': y_val}
test_data = {'text': X_test, 'label': y_test}

# Преобразуем в Dataset
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# Токенизируем
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Убираем лишние колонки
train_dataset = train_dataset.remove_columns(['text'])
val_dataset = val_dataset.remove_columns(['text'])
test_dataset = test_dataset.remove_columns(['text'])

print("    Датасет готов")

# =================== 4. ЗАГРУЗКА МОДЕЛИ ===================
print("\n Загрузка RuBERT...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Используется: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "DeepPavlov/rubert-base-cased",
    num_labels=2
).to(device)

print("    RuBERT загружен")

# =================== 5. НАСТРОЙКА ОБУЧЕНИЯ ===================
print("\n Настройка обучения...")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    report_to="none",
    dataloader_pin_memory=False,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}

# =================== 6. ОБУЧЕНИЕ ===================
print("\n Обучение RuBERT...")
print("    Это займёт 20-40 минут...")

start_time = time.time()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

train_time = time.time() - start_time
print(f"    Модель обучена за {train_time:.2f} сек ({train_time/60:.2f} мин)")

# =================== 7. СОХРАНЕНИЕ МОДЕЛИ ===================
print("\n Сохранение модели...")

os.makedirs('models', exist_ok=True)
model.save_pretrained('models')
tokenizer.save_pretrained('models')

print("    Модель сохранена в models/")

# =================== 8. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===================
print("\n Оценка модели на тестовой выборке...")

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

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

# =================== 9. CONFUSION MATRIX ===================
print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['work', 'supply'],
            yticklabels=['work', 'supply'])
plt.title('Confusion Matrix: RuBERT (base)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=150)
print("    Confusion Matrix сохранён: confusion_matrix.png")

# =================== 10. ИТОГ ===================
print("\n" + "=" * 60)
print(" ИТОГИ МОДЕЛИ 7")
print("=" * 60)
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Время обучения: {train_time:.2f} сек ({train_time/60:.2f} мин)")
print("=" * 60)

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
    f.write(f"Время обучения: {train_time:.2f} сек\n")

print("    Результаты сохранены в results.txt")
