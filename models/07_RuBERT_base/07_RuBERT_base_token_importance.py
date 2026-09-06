#!/usr/bin/env python3
"""
Визуализация важности токенов с помощью Captum (Integrated Gradients)
Автоматически определяет индекс класса Work
"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

# Путь к модели
model_path = "./models"

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Модель загружена.")

# -------------------------------------------------------------------
# 1. Автоматически определяем, какой индекс соответствует классу Work
# -------------------------------------------------------------------
# Возьмём заведомо "рабочий" текст
test_text = "Капитальный ремонт школы"
test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    logits = model(test_inputs.input_ids).logits
    work_class_index = torch.argmax(logits, dim=1).item()

print(f"Индекс класса Work (определён автоматически): {work_class_index}")
print(f"  (0 - Supply, 1 - Work)" if work_class_index == 1 else f"  (0 - Work, 1 - Supply)")

# -------------------------------------------------------------------
# 2. Текст для анализа (смешанный)
# -------------------------------------------------------------------
text = "Поставка шпаклевки с дальнейшим ремонтом учебного заведения"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
input_ids = inputs['input_ids']
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"\nТекст: {text}")
print(f"Токены: {tokens}")

# -------------------------------------------------------------------
# 3. Функция для Captum
# -------------------------------------------------------------------
def forward_func(inputs):
    return model(inputs).logits

ref_token_id = tokenizer.pad_token_id
ref_input_ids = torch.full_like(input_ids, ref_token_id)

lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

# Используем определённый индекс класса Work
attributions, delta = lig.attribute(
    input_ids,
    baselines=ref_input_ids,
    target=work_class_index,   # <--- автоматически определённый индекс
    return_convergence_delta=True
)

# Суммируем по размерности эмбеддинга
attributions = attributions.sum(dim=2).squeeze(0).cpu().detach().numpy()
valid_len = len(tokens)
attributions = attributions[:valid_len]

# -------------------------------------------------------------------
# 4. Визуализация
# -------------------------------------------------------------------
plt.figure(figsize=(14, 6))

colors = ['green' if a > 0 else 'red' for a in attributions]
plt.bar(range(len(tokens)), attributions, color=colors, alpha=0.7)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=11)
plt.xlabel('Токены', fontsize=12)
plt.ylabel('Вклад в класс Work', fontsize=12)
plt.title(f'Важность токенов для классификации\nТекст: "{text}"', fontsize=14)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3)

# Подписи значений для значимых токенов
for i, (token, val) in enumerate(zip(tokens, attributions)):
    if abs(val) > 0.5:
        plt.text(i, val + (0.1 if val > 0 else -0.3), f'{val:.2f}',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig('token_importance.png', dpi=300)
print("\n✅ Визуализация сохранена: token_importance.png")
plt.show()

# -------------------------------------------------------------------
# 5. Вывод в консоль (для понимания)
# -------------------------------------------------------------------
print("\n📊 Вклад токенов в класс Work (положительный = за Work, отрицательный = против):")
for token, val in zip(tokens, attributions):
    direction = "ЗА Work" if val > 0 else "ПРОТИВ Work"
    print(f"  {token:12} → {val:8.4f}  ({direction})")
