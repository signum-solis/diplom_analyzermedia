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
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Настройка кодировки
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Загрузка стоп-слов
nltk.download('stopwords', quiet=True)

# Путь для сохранения HTML отчетов
HTML_REPORTS_PATH = r"C:\Users\danka\IdeaProjects\Bear-02\src\main\resources\org\example\bear02"

def ensure_html_reports_dir_exists():
    """Создает директорию для отчетов, если она не существует"""
    try:
        os.makedirs(HTML_REPORTS_PATH, exist_ok=True)
    except Exception as e:
        print(f"Ошибка создания директории для отчетов: {e}", file=sys.stderr)

def save_plot_to_base64(fig):
    """Сохраняет график в base64 для вставки в HTML"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    return base64.b64encode(img.read()).decode()

def create_network_graph(data, keyword):
    """Создает граф связности тем и ключевых слов"""
    try:
        # Создаем граф
        G = nx.Graph()

        # Добавляем узлы для тем
        for topic in data['topics_info']['top_topics']:
            topic_id = topic['Topic']
            topic_name = topic['Name']
            G.add_node(f"Тема {topic_id}",
                       size=10,
                       title=topic_name,
                       group=1)

            # Добавляем примеры как связанные узлы
            for i, example in enumerate(data['topic_examples'][str(topic_id)]):
                example_id = f"Пример_{topic_id}_{i}"
                G.add_node(example_id,
                          size=5,
                          title=example[:200],
                          group=2)
                G.add_edge(f"Тема {topic_id}", example_id, weight=2)

        # Визуализация с pyvis
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)

        # Настройка физики графа
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -80000,
              "centralGravity": 0.3,
              "springLength": 200
            },
            "minVelocity": 0.75
          }
        }
        """)

        # Сохраняем граф в HTML строку
        graph_path = os.path.join(HTML_REPORTS_PATH, f"network_graph_{keyword}.html")
        net.save_graph(graph_path)
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_html = f.read()

        os.remove(graph_path)
        return graph_html

    except Exception as e:
        print(f"Ошибка создания графа: {e}", file=sys.stderr)
        return ""

def create_interactive_sentiment_chart(data):
    """Создает интерактивную диаграмму тональности с Plotly"""
    sentiments = data['sentiment_distribution']
    fig = px.pie(
        names=list(sentiments.keys()),
        values=list(sentiments.values()),
        title='Распределение тональности',
        color=list(sentiments.keys()),
        color_discrete_map={
            'Позитивная': '#4CAF50',
            'Нейтральная': '#FFC107',
            'Негативная': '#F44336'
        }
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Количество: %{value}<br>Процент: %{percent}"
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        showlegend=False
    )
    return fig.to_html(full_html=False)

def create_top_sources_chart(data):
    """Создает интерактивный график топ источников"""
    sources = {k: v for k, v in sorted(data['sources_count'].items(),
              key=lambda item: item[1], reverse=True)[:10]}

    fig = px.bar(
        x=list(sources.values()),
        y=list(sources.keys()),
        orientation='h',
        title='Топ 10 источников по количеству упоминаний',
        color=list(sources.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title="Количество упоминаний",
        yaxis_title="Источник",
        coloraxis_showscale=False
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Количество: %{x}"
    )
    return fig.to_html(full_html=False)

def create_topic_distribution_chart(data):
    """Создает график распределения тем"""
    topics_dist = data['topics_info']['topics_distribution']
    topics_info = {str(t['Topic']): t['Name'] for t in data['topics_info']['top_topics']}

    labels = [topics_info.get(k, f"Тема {k}") for k in topics_dist.keys()]

    fig = px.bar(
        x=labels,
        y=list(topics_dist.values()),
        title='Распределение упоминаний по темам',
        color=labels,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        xaxis_title="Тема",
        yaxis_title="Количество упоминаний",
        showlegend=False
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Количество: %{y}"
    )
    return fig.to_html(full_html=False)

def create_html_report(data, keyword, language):
    """Создает HTML отчет с интерактивными элементами"""
    try:
        # Убедимся, что директория для отчетов существует
        ensure_html_reports_dir_exists()

        # Создаем графики
        sentiment_chart = create_interactive_sentiment_chart(data)
        sources_chart = create_top_sources_chart(data)
        network_graph = create_network_graph(data, keyword)
        topic_chart = create_topic_distribution_chart(data)

        # Формируем HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Аналитический отчет: {keyword}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .section {{
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            width: 100%;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .metric {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 5px;
            min-width: 200px;
            text-align: center;
            flex-grow: 1;
        }}
        .metric h3 {{
            margin-top: 0;
            color: white;
        }}
        .topic-card {{
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 15px;
        }}
        .topic-examples {{
            font-style: italic;
            color: #555;
            margin-left: 15px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Аналитический отчет: {keyword.upper()}</h1>
        <p>Язык анализа: {language} | Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <h3>Всего упоминаний</h3>
            <p>{data['total_mentions']}</p>
        </div>
        <div class="metric">
            <h3>Основных тем</h3>
            <p>{data['topics_info']['total_topics']}</p>
        </div>
        <div class="metric">
            <h3>Преобладающая тональность</h3>
            <p>{max(data['sentiment_distribution'].items(), key=lambda x: x[1])[0]}</p>
        </div>
        <div class="metric">
            <h3>Основной источник</h3>
            <p>{max(data['sources_count'].items(), key=lambda x: x[1])[0]}</p>
        </div>
    </div>

    <div class="section">
        <h2>Обзор данных</h2>
        <div class="grid">
            <div class="chart-container">
                {sentiment_chart}
            </div>
            <div class="chart-container">
                {sources_chart}
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Тематический анализ</h2>
        <div class="chart-container">
            {topic_chart}
        </div>

        <h3>Связи между темами и примерами</h3>
        <div class="chart-container full-width">
            {network_graph}
        </div>

        <h3>Подробное описание тем</h3>
        {"".join([
            f"""
            <div class="topic-card">
                <h4>Тема {topic['Topic']}: {topic['Name']}</h4>
                <p>Количество упоминаний: {data['topics_info']['topics_distribution'].get(str(topic['Topic']), 0)}</p>
                <p class="topic-examples">Примеры:</p>
                <ul>
                    {"".join([f"<li>{ex[:200]}{'...' if len(ex) > 200 else ''}</li>"
                     for ex in data['topic_examples'][str(topic['Topic'])]])}
                </ul>
            </div>
            """ for topic in data['topics_info']['top_topics']
        ])}
    </div>

    <div class="section">
        <h2>Рекомендации</h2>
        <ol>
            <li><strong>Увеличить присутствие в источниках с положительной тональностью:</strong>
                Сфокусируйтесь на источниках, которые чаще публикуют позитивные материалы.</li>
            <li><strong>Разработать стратегию работы с негативными упоминаниями:</strong>
                Анализ показал {data['sentiment_distribution']['Негативная']} негативных упоминаний,
                что требует разработки плана реагирования.</li>
            <li><strong>Усилить мониторинг ключевых источников:</strong>
                Особое внимание следует уделить {list(data['sources_count'].keys())[:3]}.</li>
            <li><strong>Создать тематический контент-план:</strong>
                Основные темы обсуждения: {', '.join([t['Name'] for t in data['topics_info']['top_topics']])}.</li>
        </ol>
    </div>

    <div class="footer">
        <p>Отчет сгенерирован автоматически. Данные актуальны на {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
</body>
</html>
        """

        report_name = f"report_interactive_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        report_path = os.path.join(HTML_REPORTS_PATH, report_name)

        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML отчет сохранен как {report_path}")
        return True

    except Exception as e:
        print(f"Ошибка создания HTML отчета: {e}", file=sys.stderr)
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
    if len(sys.argv) != 3:
        print("Использование: python generate_html.py <ключевое_слово> <язык>")
        print("Доступные языки: русский, английский")
        return

    keyword = sys.argv[1]
    language = sys.argv[2].lower()

    if language not in ["русский", "английский"]:
        print("Поддерживаются только русский и английский языки")
        return

    print(f"Анализ данных для '{keyword}' на языке {language}...")
    data = analyze_data(keyword, language)

    if not data:
        print("Не удалось проанализировать данные.")
        return

    create_html_report(data, keyword, language)

if __name__ == "__main__":
    main()