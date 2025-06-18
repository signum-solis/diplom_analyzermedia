import sys
import json
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import seaborn as sns
import pandas as pd
import os
import io
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from matplotlib import rcParams
import numpy as np

# Настройка кодировки
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

REPORTS_DIR = r"C:\Users\danka\IdeaProjects\Bear-02\src\main\resources\org\example\bear02"

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

# Загрузка стоп-слов
nltk.download('stopwords', quiet=True)

class PDFReport(FPDF):
    def __init__(self, font_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.font_path = font_path or os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
        self.add_fonts()
        self.set_auto_page_break(auto=True, margin=15)

    def add_fonts(self):
        font_dir = r"C:\Users\danka\Desktop\dejavu-fonts-ttf-2.37\dejavu-fonts-ttf-2.37\ttf"
        try:
            # Основные шрифты
            self.add_font("DejaVuSans", "", os.path.join(font_dir, "DejaVuSans.ttf"), uni=True)
            self.add_font("DejaVuSans-Bold", "", os.path.join(font_dir, "DejaVuSans-Bold.ttf"), uni=True)
            self.add_font("DejaVuSans-Oblique", "", os.path.join(font_dir, "DejaVuSans-Oblique.ttf"), uni=True)

            # Серрифные шрифты (если используются)
            self.add_font("DejaVuSerif", "", os.path.join(font_dir, "DejaVuSerif.ttf"), uni=True)
            self.add_font("DejaVuSerif-Bold", "", os.path.join(font_dir, "DejaVuSerif-Bold.ttf"), uni=True)

            print("Шрифты успешно загружены")
        except Exception as e:
            print(f"Ошибка загрузки шрифтов: {e}")
            # Fallback на стандартные шрифты
            self.add_font("Arial", "", "arial.ttf", uni=True)

    def header(self):
        self.set_font('DejaVuSans-Bold', '', 12)
        self.cell(0, 10, 'Аналитический отчет по медиаданным', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVuSans-Oblique', '', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

def load_json_safe(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка загрузки файла {filepath}: {str(e)}", file=sys.stderr)
        return []

def find_data_files(language):
    """Поиск файлов с данными в разных возможных местах"""
    # Основные директории для поиска
    possible_locations = [
        os.path.dirname(__file__),  # Текущая директория скрипта
        os.path.join(os.path.dirname(__file__), "data"),
        os.path.join(os.path.dirname(__file__), "..", "data"),
        os.getcwd(),
        os.path.join(os.getcwd(), "data"),
        "C:/Users/danka/IdeaProjects/Bear-02",  # Явно добавляем вашу директорию
        "C:/Users/danka/IdeaProjects/Bear-02/data"  # И возможную поддиректорию data
    ]

    files = []
    for location in possible_locations:
        try:
            if language == "русский":
                ru_files = ["combined_news.json", "gov_news.json", "vk_news.json"]
                for f in ru_files:
                    full_path = os.path.join(location, f)
                    if os.path.exists(full_path):
                        files.append(full_path)
            else:
                en_files = ["gov.json", "news.json"]
                for f in en_files:
                    full_path = os.path.join(location, f)
                    if os.path.exists(full_path):
                        files.append(full_path)
        except Exception as e:
            print(f"Ошибка при проверке {location}: {e}", file=sys.stderr)
            continue

    return list(set(files))  # Удаляем дубликаты

def sentiment_analysis(texts, language):
    try:
        if language == "русский":
            model_name = "blanchefort/rubert-base-cased-sentiment"
        else:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512
        )

        results = []
        for text in texts:
            try:
                result = classifier(text[:4000])[0]
                results.append(result)
            except Exception as e:
                print(f"Ошибка анализа текста: {e}", file=sys.stderr)
                results.append({'label': 'NEUTRAL', 'score': 0})

        def norm_label(label):
            label = label.lower()
            if label in ['positive', 'позитивный']:
                return "Позитивная"
            elif label in ['neutral', 'нейтральный']:
                return "Нейтральная"
            else:
                return "Негативная"

        return [norm_label(r['label']) for r in results]
    except Exception as e:
        print(f"Ошибка инициализации анализа тональности: {e}", file=sys.stderr)
        return ["Нейтральная"] * len(texts)

def topic_modeling(texts, language):
    try:
        filtered_texts = [str(text) for text in texts if str(text).strip()]
        if not filtered_texts:
            print("Нет текстов для анализа!", file=sys.stderr)
            return None, []

        if language == "русский":
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            embeddings = model.encode(filtered_texts, show_progress_bar=False)

            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
            hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean',
                                  cluster_selection_method='eom', prediction_data=True)
            vectorizer_model = CountVectorizer(stop_words=stopwords.words('russian'))

            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language=None,
                nr_topics=5,
                min_topic_size=5,
                verbose=False
            )
        else:
            vectorizer_model = CountVectorizer(stop_words="english")
            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                language="english",
                nr_topics=5,
                min_topic_size=5,
                verbose=False
            )
            embeddings = None

        topics, _ = topic_model.fit_transform(filtered_texts, embeddings)
        print(f"Найдено тем: {len(set(topics)) - (1 if -1 in topics else 0)}")
        return topic_model, topics
    except Exception as e:
        print(f"Ошибка тематического моделирования: {e}", file=sys.stderr)
        return None, []

def analyze_data(keyword, language):
    # Поиск файлов с данными
    json_files = find_data_files(language)

    if not json_files:
        print(f"Не найдены файлы с данными для языка '{language}'. Искали в:", file=sys.stderr)
        print(f"- {os.path.join(os.path.dirname(__file__))}", file=sys.stderr)
        print(f"- {os.getcwd()}", file=sys.stderr)
        return None

    print(f"Найдены файлы с данными: {json_files}")

    # Загрузка данных
    all_data = []
    for jf in json_files:
        data = load_json_safe(jf)
        if data:
            all_data.extend(data)

    if not all_data:
        print("Файлы найдены, но не содержат данных для анализа", file=sys.stderr)
        return None

    # Создаем DataFrame
    try:
        df = pd.DataFrame(all_data)
        df.fillna("", inplace=True)

        # Подготовка текста
        if 'normalized_text' in df.columns:
            df['processed_text'] = df['normalized_text']
        elif 'title' in df.columns:
            df['processed_text'] = df['title']
        else:
            df['processed_text'] = df.iloc[:, 0]  # Берем первую колонку

        df['source'] = df.get('source', 'unknown')

        # Анализ тональности
        print("Анализ тональности...")
        texts = df['processed_text'].tolist()
        if not texts or all(not str(t).strip() for t in texts):
            print("Нет текста для анализа тональности", file=sys.stderr)
            return None

        df['sentiment'] = sentiment_analysis(texts, language)

        # Тематическое моделирование
        print("Тематическое моделирование...")
        topic_model, topics = topic_modeling(df['processed_text'].tolist(), language)
        if topic_model is not None:
            df['topic'] = topics[:len(df)]

        # Подготовка результатов
        metrics = {
            'total_mentions': len(df),
            'sources_count': df['source'].value_counts().to_dict(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'keyword': keyword,
            'language': language
        }

        if topic_model is not None and 'topic' in df.columns:
            topic_info = topic_model.get_topic_info()
            metrics['topics_info'] = {
                'total_topics': len(topic_info) - 1,
                'topics_distribution': df['topic'].value_counts().to_dict(),
                'top_topics': topic_info[topic_info.Topic != -1].head(3)[['Topic', 'Name']].to_dict('records')
            }

            # Добавляем примеры текстов для каждой темы
            metrics['topic_examples'] = {}
            for topic_id in df['topic'].unique():
                if topic_id != -1:  # Исключаем шум
                    examples = df[df['topic'] == topic_id]['processed_text'].head(3).tolist()
                    metrics['topic_examples'][str(topic_id)] = examples

        return metrics

    except Exception as e:
        print(f"Ошибка обработки данных: {e}", file=sys.stderr)
        return None

def create_pyramid_report(data, keyword, language):
    try:
        pdf = PDFReport()
        pdf.add_page()

        # Заголовок
        pdf.set_font("DejaVuSerif-Bold", '', 18)
        pdf.set_text_color(31, 73, 125)  # Синий цвет
        pdf.cell(0, 15, f"АНАЛИТИЧЕСКИЙ ОТЧЕТ: {keyword.upper()}", 0, 1, 'C')
        pdf.ln(10)

        # Введение
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.set_text_color(0, 0, 0)  # Черный цвет
        pdf.cell(0, 10, "ВВЕДЕНИЕ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)
        intro_text = (
            f"Настоящий отчет представляет собой анализ медиаданных по теме '{keyword}'. "
            "Отчет выполнен в формате пирамиды Минто, который включает ключевые выводы, "
            "подробный анализ и рекомендации. Данные собраны из различных источников, "
            "включая новостные агрегаторы, официальные источники и социальные сети."
        )
        pdf.multi_cell(0, 7, intro_text)
        pdf.ln(10)

        # Основные показатели
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "ОСНОВНЫЕ ПОКАЗАТЕЛИ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        # Красивые карточки с показателями
        pdf.set_fill_color(240, 248, 255)  # Светло-голубой фон
        pdf.cell(90, 30, f"Всего упоминаний: {data['total_mentions']}", 1, 0, 'C', True)
        pdf.cell(90, 30, f"Язык анализа: {language}", 1, 1, 'C', True)
        pdf.ln(10)

        # Графики
        plt.close('all')

        # График тональности (красивый donut chart)
        sentiments = data['sentiment_distribution']
        fig, ax = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax.pie(
            sentiments.values(),
            labels=sentiments.keys(),
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            colors=['#4CAF50', '#FFC107', '#F44336']
        )

        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        ax.set_title('Распределение тональности', pad=20, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('temp_sentiment.png', dpi=300, bbox_inches='tight', transparent=True)
        pdf.image('temp_sentiment.png', x=50, y=pdf.get_y(), w=100)
        pdf.ln(70)

        # График источников (горизонтальный bar chart с градиентом)
        sources = {k: v for k, v in sorted(data['sources_count'].items(),
                  key=lambda item: item[1], reverse=True)[:5]}

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(
            list(sources.keys()),
            list(sources.values()),
            color='#2196F3',
            height=0.6
        )

        # Добавляем градиент
        for bar in bars:
            bar.set_hatch('///')
            bar.set_alpha(0.8)

        ax.set_title('Топ 5 источников по количеству упоминаний', pad=15, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

        # Добавляем значения на бары
        for i, v in enumerate(sources.values()):
            ax.text(v + max(sources.values())*0.01, i, str(v), color='black', va='center')

        plt.tight_layout()
        plt.savefig('temp_sources.png', dpi=300, bbox_inches='tight', transparent=True)
        pdf.image('temp_sources.png', x=30, y=pdf.get_y(), w=140)
        pdf.ln(80)

        # Ситуация/Проблема
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "СИТУАЦИЯ И ПРОБЛЕМА", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        dominant_sentiment = max(data['sentiment_distribution'].items(), key=lambda x: x[1])[0]
        dominant_source = max(data['sources_count'].items(), key=lambda x: x[1])[0]

        situation_text = (
            f"Анализ показал, что тема '{keyword}' вызывает значительный резонанс в медиапространстве. "
            f"Преобладающей тональностью обсуждения является {dominant_sentiment.lower()}. "
            f"Основным источником информации выступает {dominant_source}. "
            "Это создает определенные вызовы и возможности для коммуникационной стратегии."
        )
        pdf.multi_cell(0, 7, situation_text)
        pdf.ln(10)

        # Рекомендации
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "РЕКОМЕНДАЦИИ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        recommendations = [
            "Увеличить присутствие в источниках с положительной тональностью",
            "Разработать стратегию работы с негативными упоминаниями",
            "Усилить мониторинг ключевых источников",
            "Создать контент-план для поддержания нейтральной/позитивной повестки",
            "Провести анализ аудитории основных источников для точечного воздействия"
        ]

        for i, rec in enumerate(recommendations):
            pdf.set_fill_color(240, 248, 255)  # Светло-голубой фон
            pdf.cell(10, 8, f"{i+1}.", 0, 0, 'R')
            pdf.multi_cell(0, 8, rec)
            pdf.ln(2)

        # Сохраняем PDF
        report_name = f"report_pyramid_.pdf"
        full_path = os.path.join(REPORTS_DIR, report_name)
        pdf.output(full_path)
        print(f"Файл сохранен: {full_path}")

    except Exception as e:
        print(f"Ошибка создания отчета: {e}", file=sys.stderr)
        return False
    return True

def create_thematic_report(data, keyword, language):
    try:
        pdf = PDFReport()
        pdf.add_page()

        # Заголовок
        pdf.set_font("DejaVuSerif-Bold", '', 18)
        pdf.set_text_color(31, 73, 125)  # Синий цвет
        pdf.cell(0, 15, f"ТЕМАТИЧЕСКИЙ ОТЧЕТ: {keyword.upper()}", 0, 1, 'C')
        pdf.ln(10)

        # Введение
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "ОСНОВНЫЕ ТЕМЫ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        if 'topics_info' in data:
            topics_info = data['topics_info']
            intro_text = (
                f"Анализ выявил {topics_info['total_topics']} основных тем в обсуждении '{keyword}'. "
                "Ниже представлены наиболее значимые темы, их характеристика и примеры контента. "
                "Данный анализ позволяет понять ключевые аспекты обсуждения темы в медиапространстве."
            )
            pdf.multi_cell(0, 7, intro_text)
            pdf.ln(10)

            # Топ темы
            pdf.set_font("DejaVuSans-Bold", '', 14)
            pdf.cell(0, 10, "ТОП ТЕМЫ", 0, 1)

            for topic in topics_info['top_topics']:
                topic_id = topic['Topic']
                topic_name = topic['Name']
                count = data['topics_info']['topics_distribution'].get(str(topic_id), 0)

                # Заголовок темы
                pdf.set_font("DejaVuSans-Bold", '', 12)
                pdf.set_text_color(31, 73, 125)  # Синий цвет
                pdf.cell(0, 8, f"Тема {topic_id}: {topic_name} ({count} упоминаний)", 0, 1)
                pdf.set_text_color(0, 0, 0)

                # Примеры текстов
                if 'topic_examples' in data and str(topic_id) in data['topic_examples']:
                    pdf.set_font("DejaVuSans-Oblique", '', 10)
                    pdf.cell(0, 5, "Примеры текстов:", 0, 1)
                    pdf.set_font("DejaVuSans", '', 10)

                    for example in data['topic_examples'][str(topic_id)]:
                        pdf.multi_cell(0, 5, f"• {example[:120]}{'...' if len(example) > 120 else ''}")

                    pdf.ln(3)

            pdf.ln(10)

        # Распределение тональности по темам
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "РАСПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ", 0, 1)

        # Создаем красивый stacked bar chart
        sentiments = data['sentiment_distribution']
        fig, ax = plt.subplots(figsize=(10, 5))

        colors = ['#4CAF50', '#FFC107', '#F44336']
        bars = ax.bar(
            sentiments.keys(),
            sentiments.values(),
            color=colors,
            width=0.6
        )

        # Добавляем значения на бары
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height/2,
                f"{height} ({height/sum(sentiments.values())*100:.1f}%)",
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )

        ax.set_title('Распределение тональности упоминаний', pad=15, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('temp_sentiment.png', dpi=300, bbox_inches='tight', transparent=True)
        pdf.image('temp_sentiment.png', x=10, y=pdf.get_y(), w=180)
        pdf.ln(70)

        # Рекомендации
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "        \n", 0, 1)
        pdf.cell(0, 10, "РЕКОМЕНДАЦИИ ПО ТЕМАМ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        recommendations = [
            "Разработать отдельные коммуникационные стратегии для каждой темы",
            "Уделить особое внимание темам с негативной тональностью",
            "Усилить присутствие в темах с положительной тональностью",
            "Создать тематические карты контента для каждой ключевой темы",
            "Провести углубленный анализ аудитории для каждой темы"
        ]

        for i, rec in enumerate(recommendations, 1):
            pdf.set_fill_color(240, 248, 255)  # Светло-голубой фон
            pdf.cell(10, 8, f"{i}.", 0, 0, 'R')
            pdf.multi_cell(0, 8, rec)
            pdf.ln(2)

        # Сохраняем PDF
        report_name = f"report_thematic_.pdf"
        full_path = os.path.join(REPORTS_DIR, report_name)
        pdf.output(full_path)
        print(f"Файл сохранен: {full_path}")

    except Exception as e:
        print(f"Ошибка создания тематического отчета: {e}", file=sys.stderr)
        return False
    return True

def create_chronological_report(data, keyword, language):
    try:
        pdf = PDFReport()
        pdf.add_page()

        # Заголовок
        pdf.set_font("DejaVuSerif-Bold", '', 18)
        pdf.set_text_color(31, 73, 125)  # Синий цвет
        pdf.cell(0, 15, f"ХРОНОЛОГИЧЕСКИЙ ОТЧЕТ: {keyword.upper()}", 0, 1, 'C')
        pdf.ln(10)

        # Основные показатели
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "ОСНОВНЫЕ ПОКАЗАТЕЛИ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        # Красивые карточки с показателями
        pdf.set_fill_color(240, 248, 255)  # Светло-голубой фон
        pdf.cell(90, 30, f"Всего упоминаний: {data['total_mentions']}", 1, 0, 'C', True)
        pdf.cell(90, 30, f"Язык анализа: {language}", 1, 1, 'C', True)
        pdf.ln(10)

        # Распределение по источникам
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "РАСПРЕДЕЛЕНИЕ ПО ИСТОЧНИКАМ", 0, 1)

        sources = {k: v for k, v in sorted(data['sources_count'].items(),
                  key=lambda item: item[1], reverse=True)[:10]}

        # Создаем красивый горизонтальный bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Создаем цветовую палитру
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(i/len(sources)) for i in range(len(sources))]

        bars = ax.barh(
            list(sources.keys()),
            list(sources.values()),
            color=colors,
            height=0.7
        )

        ax.set_title('Топ 10 источников по количеству упоминаний', pad=15, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)

        # Добавляем значения на бары
        for i, v in enumerate(sources.values()):
            ax.text(v + max(sources.values())*0.01, i, str(v), color='black', va='center')

        plt.tight_layout()
        plt.savefig('temp_sources.png', dpi=300, bbox_inches='tight', transparent=True)
        pdf.image('temp_sources.png', x=10, y=pdf.get_y(), w=180)
        pdf.ln(90)

        # Тональность
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "ТОНАЛЬНОСТЬ УПОМИНАНИЙ", 0, 1)

        sentiments = data['sentiment_distribution']

        # Создаем красивый pie chart с выносками
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))

        def func(pct, allvals):
            absolute = int(round(pct/100.*sum(allvals)))
            return f"{pct:.1f}%\n({absolute} упом.)"

        wedges, texts, autotexts = ax.pie(
            sentiments.values(),
            labels=sentiments.keys(),
            autopct=lambda pct: func(pct, sentiments.values()),
            startangle=90,
            colors=['#4CAF50', '#FFC107', '#F44336'],
            textprops=dict(color="black", fontsize=10),
            wedgeprops=dict(width=0.4, edgecolor='w'),
            pctdistance=0.85
        )

        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        ax.set_title('Распределение тональности упоминаний', pad=20, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('temp_sentiment_pie.png', dpi=300, bbox_inches='tight', transparent=True)
        pdf.image('temp_sentiment_pie.png', x=50, y=pdf.get_y(), w=100)
        pdf.ln(70)

        # Рекомендации
        pdf.set_font("DejaVuSans-Bold", '', 14)
        pdf.cell(0, 10, "РЕКОМЕНДАЦИИ", 0, 1)
        pdf.set_font("DejaVuSans", '', 12)

        recommendations = [
            "Увеличить частоту публикаций в наиболее активных источниках",
            "Разработать стратегию работы с негативными упоминаниями",
            "Мониторить динамику упоминаний для своевременного реагирования",
            "Создать систему алертов при резком росте упоминаний",
            "Анализировать сезонность обсуждения темы"
        ]

        for i, rec in enumerate(recommendations, 1):
            pdf.set_fill_color(240, 248, 255)  # Светло-голубой фон
            pdf.cell(10, 8, f"{i}.", 0, 0, 'R')
            pdf.multi_cell(0, 8, rec)
            pdf.ln(2)

        # Сохраняем PDF
        report_name = f"report_chrono.pdf"
        full_path = os.path.join(REPORTS_DIR, report_name)
        pdf.output(full_path)
        print(f"Файл сохранен: {full_path}")

    except Exception as e:
        print(f"Ошибка создания хронологического отчета: {e}", file=sys.stderr)
        return False
    return True

def main():
    if len(sys.argv) != 4:
        print("Использование: python generate_pdf.py <шаблон> <ключевое_слово> <язык>")
        print("Доступные шаблоны: пирамида, тематический, хронологический")
        print("Доступные языки: русский, английский")
        return

    template = sys.argv[1].lower()
    keyword = sys.argv[2]
    language = sys.argv[3].lower()

    if language not in ["русский", "английский"]:
        print("Поддерживаются только русский и английский языки")
        return

    print(f"Анализ данных для '{keyword}' на языке {language}...")
    data = analyze_data(keyword, language)

    if not data:
        print("Не удалось проанализировать данные. Проверьте:")
        print("- Наличие файлов с данными (combined_news.json, gov_news.json и др.)")
        print("- Формат данных в файлах (должен быть JSON)")
        print("- Доступность интернета для загрузки моделей анализа")
        return

    if template == "пирамида":
        if not create_pyramid_report(data, keyword, language):
            print("Ошибка при создании отчета", file=sys.stderr)
    elif template == "тематический":
        if not create_thematic_report(data, keyword, language):
            print("Ошибка при создании тематического отчета", file=sys.stderr)
    elif template == "хронологический":
        if not create_chronological_report(data, keyword, language):
            print("Ошибка при создании хронологического отчета", file=sys.stderr)
    else:
        print(f"Шаблон '{template}' не поддерживается")

if __name__ == "__main__":
    main()