import time
import json
import re
import argparse
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
        parts = date_str.split(', ')
        if len(parts) == 2:
            time_part, date_part = parts
            date_parts = date_part.strip().split()
            if len(date_parts) == 3:
                day = date_parts[0].zfill(2)
                month = months.get(date_parts[1].lower(), '01')
                year = date_parts[2]
                return f"{year}-{month}-{day}"
            else:
                return None
        elif len(parts) == 1:
            today = datetime.now().strftime("%Y-%m-%d")
            return today
        else:
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
        return ''

def get_news_data(keyword, date_from, date_to):
    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    date_from_iso = datetime.strptime(date_from, "%d.%m.%Y").strftime("%Y-%m-%d")
    date_to_iso = datetime.strptime(date_to, "%d.%m.%Y").strftime("%Y-%m-%d")

    url = (
        f"https://lenta.ru/search?"
        f"query={keyword}#size=10|sort=2|domain=1|"
        f"modified,format=yyyy-MM-dd|modified,from={date_from_iso}|modified,to={date_to_iso}"
    )
    driver.get(url)
    time.sleep(5)

    news_items = []
    date_from_dt = datetime.strptime(date_from, "%d.%m.%Y")
    total_processed = 0
    max_news = 250

    while True:
        articles = driver.find_elements(By.CSS_SELECTOR, ".card-full-news._search")
        if not articles:
            print("Нет новостей на странице.")
            break

        stop_loading = False

        for article in articles[len(news_items):]:
            if total_processed >= max_news:
                print("Достигнут лимит в 400 новостей. Остановка.")
                stop_loading = True
                break

            try:
                date_element = article.find_element(By.CSS_SELECTOR, ".card-full-news__date")
                published_raw = date_element.text.strip()
                published = format_date(published_raw)

                if not published:
                    continue

                published_dt = datetime.strptime(published, "%Y-%m-%d")

                if published_dt < date_from_dt:
                    stop_loading = True
                    print("Достигнута нижняя граница по дате. Остановка загрузки.")
                    break

                link_element = article.find_element(By.CSS_SELECTOR, ".card-full-news__title")
                title = link_element.text.strip()
                url_ = link_element.get_attribute('href')
                if url_ and url_.startswith('/'):
                    url_ = "https://lenta.ru" + url_
                elif not url_:
                    url_ = ""

                source = get_source_from_url(url_) or "lenta.ru"

                news_items.append({
                    'title': title,
                    'url': url_,
                    'published': published,
                    'normalized_text': normalize_text(title),
                    'source': source
                })

                total_processed += 1
                print(f"Обработано новостей: {total_processed} - {title}")

            except Exception as e:
                print(f"Ошибка при обработке новости: {e}")
                continue

        if stop_loading:
            break

        try:
            load_more_btn = driver.find_element(By.CSS_SELECTOR, ".loadmore__button")
            if load_more_btn.is_displayed():
                print("Загрузка дополнительных новостей...")
                load_more_btn.click()
                time.sleep(5)
            else:
                print("Кнопка 'Загрузить ещё' неактивна.")
                break
        except Exception:
            print("Кнопка 'Загрузить ещё' не найдена.")
            break

    driver.quit()
    return news_items

def save_to_json(data, filename='combined_news.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Сбор новостей с Lenta.ru')
    parser.add_argument('keyword', help='Ключевое слово для поиска')
    parser.add_argument('date_from', help='Дата начала в формате ДД.ММ.ГГГГ')
    parser.add_argument('date_to', help='Дата окончания в формате ДД.ММ.ГГГГ')
    parser.add_argument('--output', default='combined_news.json',
                      help='Имя выходного файла (по умолчанию: combined_news.json)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    try:
        # Проверка формата дат
        datetime.strptime(args.date_from, "%d.%m.%Y")
        datetime.strptime(args.date_to, "%d.%m.%Y")

        if datetime.strptime(args.date_to, "%d.%m.%Y") < datetime.strptime(args.date_from, "%d.%m.%Y"):
            print("Ошибка: дата окончания не может быть раньше даты начала.")
        else:
            news_data = get_news_data(args.keyword, args.date_from, args.date_to)
            save_to_json(news_data, args.output)
            print(f"Собрано {len(news_data)} новостей. Данные сохранены в '{args.output}'.")
    except ValueError as e:
        print(f"Ошибка формата даты: {e}. Используйте формат ДД.ММ.ГГГГ")