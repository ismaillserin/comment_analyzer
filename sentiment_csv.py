import pandas as pd
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# ✅ Model ve tokenizer yükle
model_path = "./roberta_sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Test verisi CSV'den yükle
input_path = "/Users/ismailserin/Desktop/comment_analyzer/Test_Yorumlar_.csv"
df = pd.read_csv(input_path)

# ✅ Duygu analizi fonksiyonu
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            score = probs[0][pred].item()
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        return label_map[pred], score
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return "error", 0.0

# ✅ Her yoruma analiz uygula
sentiments = []
scores = []

for review in df["review"]:
    sentiment, score = predict_sentiment(str(review))
    sentiments.append(sentiment)
    scores.append(score)

# ✅ Sonuçları dataframe'e ekle
df["sentiment"] = sentiments
df["score"] = scores

# ✅ Sonucu kaydet
output_path = "/Users/ismailserin/Desktop/comment_analyzer/all_test_reviews_predicted.csv"
df.to_csv(output_path, index=False)
print(f"✅ Tahminler tamamlandı. Sonuç kaydedildi: {output_path}")
