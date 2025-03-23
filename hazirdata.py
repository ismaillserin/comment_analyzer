import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

# ✅ Hugging Face Dataset'i yükle (tüm diller)
dataset = load_dataset("BaoLocTown/amazon-reviews-multi-all-languages", split="train")
df = pd.DataFrame(dataset)

# Sadece boş olmayan metin ve etiketleri al
df = df[df["text"].notna() & df["label"].notna()]

# Etiketleri kontrol et (0: negative, 1: neutral, 2: positive)
label_counts = df["label"].value_counts()
print("Etiket dağılımı:\n", label_counts)
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

# 7. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./roberta_sentiment",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
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

# (Opsiyonel) Eğitim sırasında bilgilendirme için callback
class PrintCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"📢 Epoch {int(state.epoch)} tamamlandı.")

# 9. Trainer'ı başlat
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrintCallback()]
)

# 10. Eğitim
print("🚀 Eğitim başlıyor...\n")
trainer.train()

# 11. Değerlendirme
print("\n✅ Eğitim tamamlandı. Model test verisi üzerinde değerlendiriliyor...")
results = trainer.evaluate()
print("📊 Değerlendirme Sonuçları:", results)

# 12. Kaydet
model.save_pretrained("./roberta_sentiment")
tokenizer.save_pretrained("./roberta_sentiment")
print("✅ Model ve tokenizer kaydedildi: ./roberta_sentiment klasörüne.")
