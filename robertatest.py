import time
import pandas as pd
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# ✅ Model ve tokenizer yükle
tokenizer = XLMRobertaTokenizer.from_pretrained("./roberta_sentiment")
model = XLMRobertaForSequenceClassification.from_pretrained("./roberta_sentiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ WebDriver kurulumu
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 ...")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ✅ Yorum temizleyici
def clean_review(text):
    text = re.sub(r"Daha fazla bilgi.*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

# ✅ Ürün linklerini çekme
def get_product_links(category_url, driver):
    driver.get(category_url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_links = []
    for link in soup.find_all("a", class_="a-link-normal"):
        href = link.get("href")
        if href and "/dp/" in href:
            product_links.append("https://www.amazon.com.tr" + href.split("?")[0])
    return list(set(product_links))

# ✅ Yorumları çekme
def get_reviews(product_url, driver, max_pages=3):
    reviews = []
    driver.get(product_url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-hook='review-body']"))
        )
    except:
        print("Sayfa yükleme hatası")
        return []

    time.sleep(5)
    page = 1
    while page <= max_pages:
        print(f"Scraping reviews from {product_url} - Page {page}...")
        try:
            more_info_button = driver.find_element(By.CSS_SELECTOR, "a.a-link-normal.a-text-normal")
            if more_info_button:
                more_info_button.click()
                time.sleep(5)
        except:
            pass

        soup = BeautifulSoup(driver.page_source, "html.parser")
        review_elements = soup.find_all("span", {"data-hook": "review-body"})
        for element in review_elements:
            raw_text = element.text.strip()
            cleaned = clean_review(raw_text)
            if cleaned and cleaned not in reviews:
                reviews.append(cleaned)

        try:
            next_button = driver.find_element(By.CLASS_NAME, "a-last")
            if "a-disabled" in next_button.get_attribute("class"):
                break
            next_button.click()
            time.sleep(3)
        except:
            break

        page += 1

    return reviews

# ✅ Modelle duygu analizi
def analyze_sentiment_custom(reviews):
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = []
    for review in reviews:
        if not review.strip():
            continue
        try:
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()
            sentiment = label_map[pred]
            results.append({
                "review": review,
                "sentiment": sentiment,
                "score": confidence
            })
        except Exception as e:
            print(f"RoBERTa error for: {review} → {e}")
            results.append({
                "review": review,
                "sentiment": "error",
                "score": 0.0
            })
    return results

# ✅ Ana fonksiyon
def main():
    start_time = time.time()
    category_url = "https://www.amazon.com.tr/gp/bestsellers/sporting-goods/13484252031/ref=zg_bs_nav_sporting-goods_1"
    driver = setup_driver()

    try:
        product_links = get_product_links(category_url, driver)
        print(f"Found {len(product_links)} products.")

        all_reviews = []
        for idx, product_link in enumerate(product_links):
            try:
                print(f"Scraping reviews for Product {idx + 1}: {product_link}...")
                reviews = get_reviews(product_link, driver)
                all_reviews.extend(reviews)
                print(f"Collected {len(reviews)} reviews.")
            except:
                print(f"Ürün {idx + 1} de hata oluştu")
                continue

        print("Performing sentiment analysis with XLM-RoBERTa...")
        sentiment_results = analyze_sentiment_custom(all_reviews)

        if sentiment_results:
            sentiment_df = pd.DataFrame(sentiment_results)
            print("\n✅ Final Data to be saved in CSV:")
            print(sentiment_df)

            sentiment_df.to_csv("amazon_reviews_roberta_sentiment2.csv", index=False)
            print("✅ Sentiment analysis complete. Results saved to 'amazon_reviews_roberta_sentiment.csv'.")
        else:
            print("❌ No reviews to save! Check scraping and translation.")

    finally:
        driver.quit()
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
