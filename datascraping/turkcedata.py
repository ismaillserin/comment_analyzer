from datasets import load_dataset
import pandas as pd

# Veri setini yükle (train split)
dataset = load_dataset("clapAI/MultiLingualSentiment", split="train")

# Sadece Türkçe yorumları filtrele
turkce_dataset = dataset.filter(lambda x: x['language'] == 'tr')

# text ve label sütunlarını içeren pandas DataFrame'e çevir
df = pd.DataFrame({
    "text": turkce_dataset["text"],
    "label": turkce_dataset["label"]
})

# CSV dosyasına kaydet
csv_dosya_adi = "turkce_yorumlar.csv"
df.to_csv(csv_dosya_adi, index=False, encoding='utf-8')

print(f"{len(df)} Türkçe yorum başarıyla '{csv_dosya_adi}' dosyasına kaydedildi.")
