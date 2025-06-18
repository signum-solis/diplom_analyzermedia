import json
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import os
import sys
import io
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

# Настройка кодировки
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Загрузка стоп-слов
nltk.download('stopwords', quiet=True)

def load_json_safe(filepath):
    if os.path.exists(filepath):
        with open(filepath, encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Ошибка чтения JSON файла: {filepath}", file=sys.stderr)
                return []
    else:
        print(f"Файл не найден: {filepath}", file=sys.stderr)
        return []

def sentiment_analysis(texts, language):
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
            truncated_text = text[:4000]
            result = classifier(truncated_text)[0]
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

def topic_modeling(texts, language):
    filtered_texts = [str(text) for text in texts if str(text).strip()]
    if not filtered_texts:
        print("Нет текстов для анализа!", file=sys.stderr)
        return None, []

    if language == "русский":
        # Русская модель с ruBERT
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
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
        # Английская модель
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            language="english",
            nr_topics=5,
            min_topic_size=5,
            verbose=False
        )
        embeddings = None

    try:
        topics, _ = topic_model.fit_transform(filtered_texts, embeddings)
        print(f"Найдено тем: {len(set(topics)) - (1 if -1 in topics else 0)}")
        return topic_model, topics
    except Exception as e:
        print(f"Ошибка тематического моделирования: {e}", file=sys.stderr)
        return None, []

def save_metrics(df, topic_model=None):
    metrics = {
        'total_mentions': len(df),
        'sources_count': df['source'].value_counts().to_dict(),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict()
    }

    if topic_model is not None and 'topic' in df.columns:
        topic_info = topic_model.get_topic_info()
        metrics['topics_info'] = {
            'total_topics': len(topic_info) - 1,
            'top_topics': topic_info[topic_info.Topic != -1].head(3)[['Topic', 'Name']].to_dict('records')
        }

    with open('metrics.txt', 'w', encoding='utf-8') as f:
        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
            else:
                f.write(f"{key}: {value}\n")

def main():
    if len(sys.argv) < 3:
        print("Использование: python script.py <тип_отчёта> <язык>", file=sys.stderr)
        print("Тип отчёта: пирамида минто, хронологический, тематический", file=sys.stderr)
        print("Язык: русский, английский", file=sys.stderr)
        sys.exit(1)

    report_type = sys.argv[1].lower()
    language = sys.argv[2].lower()

    json_files = []
    if language == "русский":
        json_files = [f for f in ["combined_news.json", "gov_news.json", "vk_news.json"] if os.path.exists(f)]
    elif language == "английский":
        json_files = [f for f in ["gov.json", "news.json"] if os.path.exists(f)]
    else:
        print("Неверный язык. Выберите 'русский' или 'английский'.", file=sys.stderr)
        sys.exit(1)

    all_data = []
    for jf in json_files:
        data = load_json_safe(jf)
        if data:
            all_data.extend(data)

    if not all_data:
        print("Нет данных для анализа", file=sys.stderr)
        sys.exit(0)

    df = pd.DataFrame(all_data)
    df.fillna("", inplace=True)
    df['processed_text'] = df.get('normalized_text', df.get('title', ''))
    df['source'] = df.get('source', 'unknown')

    print("Анализ тональности...")
    df['sentiment'] = sentiment_analysis(df['processed_text'].tolist(), language)

    print("Тематическое моделирование...")
    topic_model, topics = topic_modeling(df['processed_text'].tolist(), language)
    if topic_model is not None:
        df['topic'] = topics[:len(df)]

    print("\n=== Результаты анализа ===\n")

    # Общие метрики
    print(f"Всего публикаций: {len(df)}\n")

    # Распределение по тональности
    print("Распределение по тональности:")
    for sentiment, count in df['sentiment'].value_counts().items():
        print(f"  - {sentiment}: {count}")
    print()

    # Распределение по источникам
    print("Распределение по источникам:")
    for source, count in df['source'].value_counts().items():
        print(f"  - {source}: {count}")
    print()

    # Тематическое моделирование
    if 'topic' in df.columns:
        print("Топ темы и примеры публикаций:\n")
        topic_counts = df['topic'].value_counts().drop(-1, errors='ignore')  # убираем шум
        top_topics = topic_counts.head(3)

        for topic_id in top_topics.index:
            topic_name = topic_model.get_topic(topic_id)
            topic_words = ", ".join([word for word, _ in topic_name][:5])
            count = topic_counts[topic_id]
            print(f"Тема {topic_id} ({count} упоминаний): {topic_words}")
            sample_texts = df[df['topic'] == topic_id]['processed_text'].head(2)
            for i, txt in enumerate(sample_texts, 1):
                print(f"    Пример {i}: {txt[:100]}{'...' if len(txt) > 100 else ''}")
            print()
    else:
        print("Темы не были сгенерированы.")

    # Сохранение метрик
    save_metrics(df, topic_model)
    print("\nМетрики сохранены в metrics.txt")

if __name__ == "__main__":
    main()