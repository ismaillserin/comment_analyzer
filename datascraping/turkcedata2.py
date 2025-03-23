from datasets import load_dataset
import pandas as pd

# Veri setini yükle
dataset = load_dataset("maydogan/Turkish_SentimentAnalysis_TRSAv1", split="train")

# Etiket başına alınacak örnek sayısı
label_limits = {
    "positive": 10000,
    "negative": 7000,
    "neutral": 7000
}

# Sonuçları saklamak için liste
filtered_data = []

# Her etiket için verileri filtrele
for label, count in label_limits.items():
    subset = dataset.filter(lambda x: x['score'].strip().lower() == label)
    selected_count = min(len(subset), count)
    print(f"{label.capitalize()}: {selected_count} örnek alındı")
    filtered_data.extend(subset.select(range(selected_count)))

# DataFrame'e çevir (sadece review ve label)
df = pd.DataFrame({
    "text": [ex["review"] for ex in filtered_data],
    "label": [ex["score"].strip().lower() for ex in filtered_data]
})

# CSV olarak kaydet
output_path = "../reviews/tr_reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Toplam {len(df)} yorum '{output_path}' dosyasına kaydedildi.")
