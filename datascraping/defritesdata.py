from datasets import load_dataset
import pandas as pd

# Veri setini yükle
dataset = load_dataset("clapAI/MultiLingualSentiment", split="train")

# Hedef diller ve etiket başına istenen örnek sayıları
target_languages = ["de", "fr", "it", "es"]
label_limits = {"positive": 3000, "negative": 2500, "neutral": 2500}

# Sonuçları saklayacağımız liste
filtered_data = []

# Her dil ve her etiket için verileri filtrele
for lang in target_languages:
    print(f"\nDil: {lang}")
    for label, count in label_limits.items():
        # Filtreleme
        subset = dataset.filter(lambda x: x['language'] == lang and x['label'].strip().lower() == label)
        print(f"  {label.capitalize()}: {min(len(subset), count)} örnek alındı")
        # Seçilen verileri ekle
        filtered_data.extend(subset.select(range(min(len(subset), count))))

# Pandas DataFrame'e çevir (SADECE text ve label sütunları)
df = pd.DataFrame({
    "text": [ex["text"] for ex in filtered_data],
    "label": [ex["label"].strip().lower() for ex in filtered_data]
})

# CSV olarak kaydet
output_path = "../reviews/de_it_fr_es_reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Toplam {len(df)} yorum '{output_path}' dosyasına SADECE text ve label ile kaydedildi.")
