import time
import json
import re
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
from datetime import datetime
import argparse
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def format_date(date_str):
    months = {
        'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
        'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
        'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
    }
    try:
        parts = date_str.strip().split()
        if len(parts) == 3:
            day = parts[0].zfill(2)
            month = months.get(parts[1].lower(), '01')
            year = parts[2]
            return f"{year}-{month}-{day}"
        return None
    except Exception as e:
        print(f"Ошибка при форматировании даты '{date_str}': {e}")
        return None

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    return text.strip()

def get_source_from_url(url):
    try:
        return re.search(r'https?://([^/]+)/', url).group(1)
    except:
        return 'government.ru'

def get_news_data(date_from, date_to):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    url = f"http://government.ru/news/?dt.since={date_from}&dt.till={date_to}"
    driver.get(url)
    time.sleep(5)

    articles = driver.find_elements(By.CSS_SELECTOR, ".headline")
    news_items = []

    for article in articles:
        try:
            date_elem = article.find_element(By.CLASS_NAME, "headline_date")
            date_str = date_elem.text.strip()
            published = format_date(date_str)

            link_elem = article.find_element(By.CLASS_NAME, "headline__link")
            url_ = link_elem.get_attribute("href")

            title_elem = link_elem.find_element(By.CLASS_NAME, "headline_title_link")
            title = title_elem.text.strip()

            news_items.append({
                "title": title,
                "url": url_,
                "published": published,
                "normalized_text": normalize_text(title),
                "source": get_source_from_url(url_)
            })

            print(f"[+] {published} - {title}")

        except Exception as e:
            print(f"[!] Ошибка при обработке статьи: {e}")
            continue

    driver.quit()
    return news_items

def save_to_json(data, filename="gov_news.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    date_from = sys.argv[2]
    date_to = sys.argv[3]

    try:
        df_obj = datetime.strptime(date_from, "%d.%m.%Y")
        dt_obj = datetime.strptime(date_to, "%d.%m.%Y")

        if dt_obj < df_obj:
            print("Дата окончания не может быть раньше даты начала.")
            sys.exit(1)

        date_from_url = df_obj.strftime("%d.%m.%Y")
        date_to_url = dt_obj.strftime("%d.%m.%Y")

        news = get_news_data(date_from_url, date_to_url)
        save_to_json(news)
        print(f"\nСобрано {len(news)} новостей. Сохранено в gov_news.json.")
    except ValueError:
        print("Ошибка в формате даты. Используйте ДД.ММ.ГГГГ.")
        sys.exit(1)
