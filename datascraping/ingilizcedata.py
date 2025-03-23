from datasets import load_dataset
import pandas as pd

# Veri setini yükle
dataset = load_dataset("clapAI/MultiLingualSentiment", split="train")

# Dil: İngilizce
target_language = "en"
label_limits = {"positive": 6000, "negative": 4000, "neutral": 4000}

# Sonuçları saklayacağımız liste
filtered_data = []

print(f"\nDil: {target_language}")
for label, count in label_limits.items():
    # Filtreleme
    subset = dataset.filter(lambda x: x['language'] == target_language and x['label'].strip().lower() == label)
    selected_count = min(len(subset), count)
    print(f"  {label.capitalize()}: {selected_count} örnek alındı")
    filtered_data.extend(subset.select(range(selected_count)))

# Pandas DataFrame'e çevir (sadece text ve label sütunları)
df = pd.DataFrame({
    "text": [ex["text"] for ex in filtered_data],
    "label": [ex["label"].strip().lower() for ex in filtered_data]
})

# CSV olarak kaydet
output_path = "../reviews/en_reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Toplam {len(df)} yorum '{output_path}' dosyasına kaydedildi.")
