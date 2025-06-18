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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Для остановки "nltk" при первом запуске — раскомментируй
nltk.download('stopwords')

def load_news(path="filtered_news.json"):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def sentiment_analysis(texts):
    classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
    results = classifier(texts)
    def norm_label(label):
        if label.lower() in ['positive', 'позитивный']:
            return "Позитивная"
        elif label.lower() in ['neutral', 'нейтральный']:
            return "Нейтральная"
        else:
            return "Негативная"
    return [norm_label(r['label']) for r in results]

def topic_modeling(texts):
    russian_stopwords = stopwords.words('russian')
    vectorizer_model = CountVectorizer(stop_words=russian_stopwords)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="russian")
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics

def generate_report_text(df, topic_model):
    lines = [
        "# Сжатый аналитический отчет по новостям\n",
        "## Обзор\n",
        "Этот отчет представляет сжатый анализ новостных публикаций: ключевые темы, источники и тональность.\n",
        "## Анализ тональности\n"
    ]

    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    for sentiment in ['Позитивная', 'Нейтральная', 'Негативная']:
        val = sentiment_counts.get(sentiment, 0)
        lines.append(f"- **{sentiment}**: {val:.1f}%")

    lines.append("\n## Источники и активность\n")
    source_counts = df['source'].value_counts().head(3)
    for src, cnt in source_counts.items():
        lines.append(f"- **{src}**: {cnt} публикаций")

    lines.append("\n### Тональность по активным источникам\n")
    for src in source_counts.index:
        temp = df[df['source'] == src]
        sc = temp['sentiment'].value_counts(normalize=True) * 100
        parts = [f"{k}: {v:.0f}%" for k, v in sc.items()]
        lines.append(f"- {src}: " + ", ".join(parts))

    lines.append("\n## Основные темы\n")
    topics_info = topic_model.get_topic_info()
    top_topics = topics_info[topics_info.Topic != -1].head(3)
    for _, row in top_topics.iterrows():
        lines.append(f"- **Тема {row.Topic}**: {row.Name} ({row.Count} документов)")

    lines.append("\n## Заключение\n")
    lines.append(
        "Анализ выявил ведущие источники публикаций, их тональные различия и основные тематические кластеры, "
        "что позволяет судить о текущей повестке и эмоциональной окраске новостного поля."
    )

    return "\n".join(lines)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_console()
    news = load_news()
    df = pd.DataFrame(news)
    df.fillna("", inplace=True)

    if 'source' not in df.columns:
        df['source'] = "Неизвестный источник"

    print("Запуск анализа тональности...")
    df['sentiment'] = sentiment_analysis(df['title'].tolist())

    print("Выделение ключевых тем...")
    topic_model, topics = topic_modeling(df['title'].tolist())
    df['topic'] = topics

    report_text = generate_report_text(df, topic_model)
    print("\n--- Отчет ---\n")
    print(report_text)

if __name__ == "__main__":
    main()
