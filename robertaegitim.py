import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback

# 1. Verileri oku ve birleştir
data_dir = "reviews"

csv_files = [
    "de_it_fr_es_reviews.csv",
    "en_reviews.csv",
    "ru_ar_ja_reviews.csv",
    "tr_reviews.csv"
]

df_list = [pd.read_csv(os.path.join(data_dir, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Etiketleri dönüştür
label2id = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].str.strip().str.lower().map(label2id)

# ✅ Etiket dağılımı kontrol
print("🔍 Etiket dağılımı:")
print(df["label"].value_counts())

print(f"✅ Veri yüklendi. Toplam örnek sayısı: {len(df)}")

# 2. Eğitim ve test setine ayır
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
print(f"✅ Eğitim örnekleri: {len(train_texts)}, Test örnekleri: {len(test_texts)}")

# 3. Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# 4. Dataset sınıfı
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 5. Dataset'leri oluştur
train_dataset = CustomDataset(train_texts, train_labels)
test_dataset = CustomDataset(test_texts, test_labels)

# 6. Modeli yükle (3 sınıf için)
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)
print("✅ Model yüklendi: xlm-roberta-base")

# 7. Eğitim ayarları (güncellenmiş)
training_args = TrainingArguments(
    output_dir="./roberta_sentiment",
    num_train_epochs=6,  # ⬅️ Epoch sayısı artırıldı (4 → 6)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,
    fp16=True if torch.cuda.is_available() else False
)

# 8. Metrik fonksiyonu
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro"),
    }

# 9. Callback sınıfları
class PrintCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"📢 Epoch {int(state.epoch)} tamamlandı.")

# 10. Trainer başlat (EarlyStopping eklendi)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        PrintCallback(),
        EarlyStoppingCallback(early_stopping_patience=2)  # ⬅️ Yeni eklendi
    ]
)

# 11. Eğitim başlat
print("🚀 Eğitim başlıyor...\n")
trainer.train()

# 12. Test veri seti üzerinde değerlendirme
print("\n✅ Eğitim tamamlandı. Model test verisi üzerinde değerlendiriliyor...")
results = trainer.evaluate()
print("📊 Değerlendirme Sonuçları:", results)

# 13. Ayrıntılı sınıf bazlı analiz
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\n📊 Sınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))

print("🧱 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 14. Modeli kaydet
model.save_pretrained("./roberta_sentiment")
tokenizer.save_pretrained("./roberta_sentiment")
print("✅ Model ve tokenizer kaydedildi: ./roberta_sentiment klasörüne.")
