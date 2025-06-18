import requests
import time
import json
import re
import sys
from datetime import datetime
import argparse
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ВСТАВЬТЕ СЮДА СВОЙ СЕРВИСНЫЙ КЛЮЧ ДОСТУПА VK
ACCESS_TOKEN = ""

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    return text.strip()

def date_to_timestamp(date_str):
    return int(datetime.strptime(date_str, "%d.%m.%Y").timestamp())

def timestamp_to_date(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

def search_vk_news(query, date_from, date_to, max_results=400):
    url = "https://api.vk.com/method/newsfeed.search"
    version = "5.199"

    start_time = date_to_timestamp(date_from)
    end_time = date_to_timestamp(date_to)

    all_items = []
    count = 100
    offset = 0

    while len(all_items) < max_results:
        params = {
            "q": query,
            "access_token": ACCESS_TOKEN,
            "v": version,
            "count": count,
            "start_time": start_time,
            "end_time": end_time,
            "offset": offset
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "error" in data:
            print("Ошибка VK API:", data["error"])
            break

        items = data.get("response", {}).get("items", [])
        if not items:
            break

        for item in items:
            title = item.get("text", "").strip()
            if not title:
                continue

            news = {
                "title": title,
                "url": f"https://vk.com/wall{item['owner_id']}_{item['id']}",
                "published": timestamp_to_date(item['date']),
                "normalized_text": normalize_text(title),
                "source": "vk.com"
            }
            all_items.append(news)

            if len(all_items) >= max_results:
                break

        offset += count
        time.sleep(0.3)

    return all_items

def save_to_json(data, filename="vk_news.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Использование: python vk_script.py <ключевое_слово> <дата_начала> <дата_окончания>")
        print("Формат даты: ДД.ММ.ГГГГ")
        sys.exit(1)

    query = sys.argv[1]
    date_from = sys.argv[2]
    date_to = sys.argv[3]

    try:
        df = datetime.strptime(date_from, "%d.%m.%Y")
        dt = datetime.strptime(date_to, "%d.%m.%Y")

        if dt < df:
            print("Ошибка: дата окончания раньше даты начала.")
            sys.exit(1)

        results = search_vk_news(query, date_from, date_to)
        save_to_json(results)
        print(f"Собрано {len(results)} новостей. Данные сохранены в '_news.json'.")

    except ValueError:
        print("Ошибка в формате даты. Используйте ДД.ММ.ГГГГ.")
        sys.exit(1)
