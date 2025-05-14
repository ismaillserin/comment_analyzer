import pandas as pd
from tqdm import tqdm
from langdetect import detect

# Dosyayı oku
df = pd.read_csv("/Users/ismailserin/Desktop/comment_analyzer/amazon_reviews_roberta_sentiment7.csv")

# Dil tespiti
languages = []
for review in tqdm(df["review"], desc="Dilleri tespit ediliyor"):
    try:
        lang = detect(str(review))
    except:
        lang = "unknown"
    languages.append(lang)

df["language"] = languages

# Kullanıcıdan doğru/yanlış kontrolü iste
def manual_check(row):
    print("\nYorum:")
    print(row["review"])
    print("Model Tahmini:", row["sentiment"])
    choice = input("Doğru mu? (1 = doğru, 0 = yanlış): ")
    if choice.strip() == "1":
        return True
    else:
        return False

# Her dil için analiz yap
result = {}

for lang in df["language"].unique():
    subset = df[df["language"] == lang]
    print(f"\n--- {lang.upper()} Dilinde {len(subset)} yorum var ---")
    correct = 0
    for idx, row in subset.iterrows():
        if manual_check(row):
            correct += 1
    total = len(subset)
    accuracy = correct / total if total > 0 else 0
    result[lang] = {
        "toplam": total,
        "dogru": correct,
        "yanlis": total - correct,
        "dogruluk_orani": round(accuracy * 100, 2)
    }

# Sonuçları yazdır
print("\n--- SONUÇLAR ---")
for lang, stats in result.items():
    print(f"{lang.upper()} - Toplam: {stats['toplam']}, Doğru: {stats['dogru']}, Yanlış: {stats['yanlis']}, Doğruluk Oranı: %{stats['dogruluk_orani']}")
