import sys
import json
import matplotlib.pyplot as plt
import os
import io
from datetime import datetime
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import base64

# Настройка кодировки
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Загрузка стоп-слов
nltk.download('stopwords', quiet=True)

def save_plot_to_base64(fig):
    """Сохраняет график в base64 для вставки в Markdown"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.read()).decode()

def create_pyramid_report(data, keyword, language):
    """Создает отчет в формате пирамиды Минто в Markdown"""
    try:
        # Создаем графики
        plt.close('all')

        # График тональности
        sentiments = data['sentiment_distribution']
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax1.pie(
            sentiments.values(),
            labels=sentiments.keys(),
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='w'),
            colors=['#4CAF50', '#FFC107', '#F44336']
        )
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=10)
        ax1.set_title('Распределение тональности', pad=20, fontsize=12, fontweight='bold')
        sentiment_img = save_plot_to_base64(fig1)

        # График источников
        sources = {k: v for k, v in sorted(data['sources_count'].items(),
                  key=lambda item: item[1], reverse=True)[:5]}
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars = ax2.barh(
            list(sources.keys()),
            list(sources.values()),
            color='#2196F3',
            height=0.6
        )
        ax2.set_title('Топ 5 источников по количеству упоминаний', pad=15, fontsize=12, fontweight='bold')
        for i, v in enumerate(sources.values()):
            ax2.text(v + max(sources.values())*0.01, i, str(v), color='black', va='center')
        sources_img = save_plot_to_base64(fig2)

        # Формируем Markdown
        md_content = f"""# Аналитический отчет: {keyword.upper()}

## Введение
Настоящий отчет представляет собой анализ медиаданных по теме '{keyword}'.
Отчет выполнен в формате пирамиды Минто, который включает ключевые выводы,
подробный анализ и рекомендации.

## Основные показатели
- **Всего упоминаний**: {data['total_mentions']}
- **Язык анализа**: {language}

## Распределение тональности
![Распределение тональности](data:image/png;base64,{sentiment_img})

## Топ источников
![Топ источников](data:image/png;base64,{sources_img})

## Ситуация и проблема
Преобладающей тональностью обсуждения является {max(data['sentiment_distribution'].items(), key=lambda x: x[1])[0].lower()}.
Основным источником информации выступает {max(data['sources_count'].items(), key=lambda x: x[1])[0]}.

## Рекомендации
1. Увеличить присутствие в источниках с положительной тональностью
2. Разработать стратегию работы с негативными упоминаниями
3. Усилить мониторинг ключевых источников
4. Создать контент-план для поддержания нейтральной/позитивной повестки

*Отчет сгенерирован {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

        report_name = f"report_pyramid_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Markdown отчет сохранен как {report_name}")

        return True

    except Exception as e:
        print(f"Ошибка создания отчета: {e}", file=sys.stderr)
        return False

def create_thematic_report(data, keyword, language):
    """Создает тематический отчет в Markdown"""
    try:
        # Создаем графики
        plt.close('all')

        # График тональности
        sentiments = data['sentiment_distribution']
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        colors = ['#4CAF50', '#FFC107', '#F44336']
        bars = ax1.bar(
            sentiments.keys(),
            sentiments.values(),
            color=colors,
            width=0.6
        )
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height/2,
                f"{height} ({height/sum(sentiments.values())*100:.1f}%)",
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
        ax1.set_title('Распределение тональности упоминаний', pad=15, fontsize=12, fontweight='bold')
        sentiment_img = save_plot_to_base64(fig1)

        # Формируем Markdown
        md_content = f"""# Тематический отчет: {keyword.upper()}

## Основные темы
Анализ выявил {data['topics_info']['total_topics']} основных тем в обсуждении '{keyword}'.

### Топ темы
"""

        # Добавляем информацию о темах
        for topic in data['topics_info']['top_topics']:
            topic_id = topic['Topic']
            topic_name = topic['Name']
            count = data['topics_info']['topics_distribution'].get(str(topic_id), 0)

            md_content += f"""
**Тема {topic_id}**: {topic_name} ({count} упоминаний)

Примеры:
"""
            for example in data['topic_examples'][str(topic_id)]:
                md_content += f"- {example[:120]}{'...' if len(example) > 120 else ''}\n"

        md_content += f"""
## Распределение тональности
![Распределение тональности](data:image/png;base64,{sentiment_img})

## Рекомендации по темам
1. Разработать отдельные коммуникационные стратегии для каждой темы
2. Уделить особое внимание темам с негативной тональностью
3. Усилить присутствие в темах с положительной тональностью

*Отчет сгенерирован {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

        report_name = f"report_thematic_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Markdown отчет сохранен как {report_name}")
        return True

    except Exception as e:
        print(f"Ошибка создания тематического отчета: {e}", file=sys.stderr)
        return False

def create_chronological_report(data, keyword, language):
    """Создает хронологический отчет в Markdown"""
    try:
        # Создаем графики
        plt.close('all')

        # График источников
        sources = {k: v for k, v in sorted(data['sources_count'].items(),
                  key=lambda item: item[1], reverse=True)[:10]}
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(i/len(sources)) for i in range(len(sources))]
        bars = ax1.barh(
            list(sources.keys()),
            list(sources.values()),
            color=colors,
            height=0.7
        )
        ax1.set_title('Топ 10 источников по количеству упоминаний', pad=15, fontsize=12, fontweight='bold')
        for i, v in enumerate(sources.values()):
            ax1.text(v + max(sources.values())*0.01, i, str(v), color='black', va='center')
        sources_img = save_plot_to_base64(fig1)

        # График тональности
        sentiments = data['sentiment_distribution']
        fig2, ax2 = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax2.pie(
            sentiments.values(),
            labels=sentiments.keys(),
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100.*sum(sentiments.values())))})",
            startangle=90,
            colors=['#4CAF50', '#FFC107', '#F44336'],
            textprops=dict(color="black", fontsize=10),
            wedgeprops=dict(width=0.4, edgecolor='w'),
            pctdistance=0.85
        )
        ax2.set_title('Распределение тональности упоминаний', pad=20, fontsize=12, fontweight='bold')
        sentiment_img = save_plot_to_base64(fig2)

        # Формируем Markdown
        md_content = f"""# Хронологический отчет: {keyword.upper()}

## Основные показатели
- **Всего упоминаний**: {data['total_mentions']}
- **Язык анализа**: {language}

## Распределение по источникам
![Топ источников](data:image/png;base64,{sources_img})

## Тональность упоминаний
![Распределение тональности](data:image/png;base64,{sentiment_img})

## Рекомендации
1. Увеличить частоту публикаций в наиболее активных источниках
2. Разработать стратегию работы с негативными упоминаниями
3. Мониторить динамику упоминаний для своевременного реагирования

*Отчет сгенерирован {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

        report_name = f"report_chronological_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Markdown отчет сохранен как {report_name}")
        return True

    except Exception as e:
        print(f"Ошибка создания хронологического отчета: {e}", file=sys.stderr)
        return False
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

def main():
    if len(sys.argv) != 4:
        print("Использование: python generate_md.py <шаблон> <ключевое_слово> <язык>")
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
        print("Не удалось проанализировать данные.")
        return

    if template == "пирамида минто":
        create_pyramid_report(data, keyword, language)
    elif template == "тематическая":
        create_thematic_report(data, keyword, language)
    elif template == "хронологическая":
        create_chronological_report(data, keyword, language)
    else:
        print(f"Шаблон '{template}' не поддерживается")

if __name__ == "__main__":
    main()