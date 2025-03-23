from datasets import load_dataset
import pandas as pd

# Veri setini yükle
dataset = load_dataset("clapAI/MultiLingualSentiment", split="train")

# Hedef diller ve sınıf başına örnek sayısı
target_languages = ["ru", "ar", "ja"]
label_limits = {"positive": 1500, "negative": 1000, "neutral": 1000}

# Sonuçları tutacak liste
filtered_data = []

# Her dil ve her etiket için verileri filtrele
for lang in target_languages:
    print(f"\nDil: {lang}")
    for label, count in label_limits.items():
        subset = dataset.filter(lambda x: x['language'] == lang and x['label'].strip().lower() == label)
        selected_count = min(len(subset), count)
        print(f"  {label.capitalize()}: {selected_count} örnek alındı")
        filtered_data.extend(subset.select(range(selected_count)))

# DataFrame'e çevir (sadece text ve label)
df = pd.DataFrame({
    "text": [ex["text"] for ex in filtered_data],
    "label": [ex["label"].strip().lower() for ex in filtered_data]
})

# CSV olarak kaydet
output_path = "../reviews/ru_ar_ja_reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Toplam {len(df)} yorum '{output_path}' dosyasına kaydedildi.")
