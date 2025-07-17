from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
from huggingface_hub import login

# Логін до Hugging Face з токеном (отриманим у профілі)
login(token="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")

from datasets import load_dataset

# Тепер можна завантажити датасет без помилки
imdb = load_dataset("imdb")

# 2. Завантажуємо токенізатор
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# 3. Функція для токенізації текстів з обрізкою
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# 4. Токенізуємо датасет
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# 5. Підготовка data collator для динамічного паддінгу
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Визначаємо метрику accuracy для оцінки
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 7. Завантажуємо модель для класифікації (2 класи)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 8. Встановлюємо параметри тренування
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

# 9. Ініціалізуємо тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10. Запускаємо тренування
trainer.train()

# 11. Інференс (передбачення) за допомогою pipeline
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(classifier("This was a masterpiece. Not completely faithful to the books, but enthralling…"))
