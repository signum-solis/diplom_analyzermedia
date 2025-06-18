import feedparser
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import dateparser
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import argparse
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

nltk.download('stopwords')

RSS_FEEDS = [
    "https://www.interfax.ru/rss.asp",
    "https://ria.ru/export/rss2/news/index.xml",               # РИА Новости
    "https://tass.ru/rss/v2.xml",                               # ТАСС
    "https://www.vedomosti.ru/rss/news",                        # Ведомости
    "https://www.mk.ru/rss/all_news.xml",                       # Московский Комсомолец
    "https://www.kommersant.ru/RSS/news.xml",                   # Коммерсантъ
    "https://tvzvezda.ru/export/rss.xml",                       # ТВ Звезда
    "https://iz.ru/xml/rss/all.xml",                            # Известия
    "https://www.gazeta.ru/export/rss/lenta.xml",               # Газета.Ru
    "https://argumenti.ru/rss.xml",                             # Аргументы.ру
    "https://www.fontanka.ru/fontanka.rss",                     # Фонтанка (СПб)
    "https://www.novayagazeta.ru/rss/all.xml",                  # Новая Газета
    "https://svpressa.ru/rss/all.xml",                          # Свободная Пресса
    "https://www.ridus.ru/export/rss.xml",                      # Ридус
    "https://360tv.ru/rss/",                                    # 360° Подмосковье
    "https://echo.msk.ru/interview/rss-fulltext.xml"
]

STOPWORDS = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")

DEBUG_MODE = True

# --- Логгер ---
class Logger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        self.logfile = open(logfile_path, 'a', encoding='utf-8')

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)  # В консоль
        self.logfile.write(full_msg + '\n')  # В файл
        self.logfile.flush()

    def close(self):
        self.logfile.close()

# --- Инициализируем логгер (путь к лог файлу) ---
logger = Logger("logs/user_actions_log.txt")

def clean_and_stem_text(text):
    tokens = re.findall(r'\b[а-яё]+\b', text.lower())
    filtered = [stemmer.stem(word) for word in tokens if word not in STOPWORDS]
    return filtered


def extract_dates_from_text(text, fallback_year):
    dates = []
    date_patterns = [
        r'\d{1,2}\s+[а-яё]+(?:\s+\d{4})?',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}\.\d{1,2}\.\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}',
    ]

    for pattern in date_patterns:
        for match in re.findall(pattern, text.lower()):
            dt = dateparser.parse(match, languages=['ru'])
            if dt:
                dates.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

    dt = dateparser.parse(text, languages=['ru'])
    if dt:
        dates.append(dt.replace(hour=0, minute=0, second=0, microsecond=0))

    return dates


def date_in_range_loosely(dates_list, start_dt, end_dt):
    for dt in dates_list:
        if start_dt <= dt <= end_dt:
            return True
        if start_dt.year <= dt.year <= end_dt.year:
            if start_dt.year == end_dt.year:
                if start_dt.month <= dt.month <= end_dt.month:
                    return True
            else:
                return True
    return False


def get_dates_from_page(url, fallback_year):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        texts = soup.stripped_strings
        full_text = ' '.join(texts)
        dates = extract_dates_from_text(full_text, fallback_year)
        if DEBUG_MODE:
            print(f"Найдено дат на странице: {[d.strftime('%Y-%m-%d') for d in dates]}")
        return dates
    except Exception as e:
        if DEBUG_MODE:
            print(f"Ошибка при загрузке страницы {url}: {e}")
        return []


def fetch_filtered_news(keyword, start_date, end_date):
    all_news = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    keyword_stem = stemmer.stem(keyword.lower())

    for feed_url in RSS_FEEDS:
        source_name = feed_url.split("//")[-1].split("/")[0]
        print(f"\nЧтение ленты: {feed_url} (Источник: {source_name})")
        feed = feedparser.parse(feed_url)
        print(f"Найдено статей: {len(feed.entries)}")

        for entry in feed.entries:
            title = entry.get('title', '')
            link = entry.get('link', '')
            summary = entry.get('summary', '') or ''
            full_text = f"{title} {summary}"
            normalized_words = clean_and_stem_text(full_text)

            if keyword_stem not in normalized_words:
                if DEBUG_MODE:
                    print(f"Нет ключевого слова '{keyword}' в статье: {title}")
                continue

            dates_found = get_dates_from_page(link, fallback_year=start_dt.year)
            if not dates_found:
                if DEBUG_MODE:
                    print(f"Даты не найдены на странице: {title}")
                continue

            if not date_in_range_loosely(dates_found, start_dt, end_dt):
                if DEBUG_MODE:
                    print(f"Дата вне диапазона: {title}")
                continue

            published_dt = max(dates_found)

            print(f"Найдена новость: {title} [{published_dt.strftime('%Y-%m-%d')}]")

            all_news.append({
                "title": title,
                "url": link,
                "published": published_dt.strftime('%Y-%m-%d'),
                "normalized_text": ' '.join(normalized_words),
                "source": source_name
            })

    return all_news


def main():
    parser = argparse.ArgumentParser(description='Новости по ключевому слову и дате')
    parser.add_argument('keyword', type=str, help='Ключевое слово для поиска')
    parser.add_argument('start_date', type=str, help='Начальная дата в формате ГГГГ-ММ-ДД')
    parser.add_argument('end_date', type=str, help='Конечная дата в формате ГГГГ-ММ-ДД')
    args = parser.parse_args()

    filtered_news = fetch_filtered_news(args.keyword, args.start_date, args.end_date)

    filename = "filtered_news.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(filtered_news, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено {len(filtered_news)} новостей в файл {filename}")

    logger.log(f"Сохранено {len(filtered_news)} новостей в файл {filename}")

    logger.close()


if __name__ == "__main__":
    main()
