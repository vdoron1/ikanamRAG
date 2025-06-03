import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image

def filter_generic_texts(df, phrases):
    mask = df['chunk_text'].str.lower().apply(
        lambda text: not any(phrase in text for phrase in phrases)
    )
    return df[mask]

logo = Image.open('photo_2024-11-11_21-00-18.jpg')
st.image(logo, width=150)

@st.cache_data
def load_data():
    df = pd.read_csv('ikanam_chunks.csv')
    if 'year' not in df.columns:
        df['year'] = df['title'].str.extract(r'\((\d{4})')[0].astype('Int64')
    if 'chunk_uid' not in df.columns:
        df['chunk_uid'] = df['document_id'].astype(str) + '_' + df['chunk_id'].astype(str)
    if 'source_url' not in df.columns:
        df['source_url'] = "https://vk.com/ikanam"
    return df

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=True)
def load_faiss_index(index_path: str, embeddings_path: str):
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    return index, embeddings

@st.cache_data
def generate_wordcloud(text):
    return WordCloud(width=800, height=400, background_color='white').generate(text)

# Загрузка данных и моделей
df_chunks = load_data()
model = load_model()
index, embeddings = load_faiss_index('ikanam_faiss.index', 'ikanam_embeddings.npy')

st.title("ikanamRAG — Анализ, визуализация и RAG-поиск")

# Общая статистика
st.header("Общая статистика")
st.write(f"Уникальных документов: {df_chunks['document_id'].nunique()}")
st.write(f"Всего чанков: {len(df_chunks)}")

# Фильтр по типу чанков
comment_filter = st.multiselect(
    "Выберите типы чанков",
    options=df_chunks['comment'].unique(),
    default=df_chunks['comment'].unique()
)
df_filtered = df_chunks[df_chunks['comment'].isin(comment_filter)].copy()

# Фильтр по годам
years = sorted(df_filtered['year'].dropna().unique())
year_selected = st.slider(
    "Выберите диапазон годов",
    min_value=years[0],
    max_value=years[-1],
    value=(years[0], years[-1])
)
df_filtered = df_filtered[(df_filtered['year'] >= year_selected[0]) & (df_filtered['year'] <= year_selected[1])]

# RAG-поиск
query = st.text_input("Введите запрос для поиска по корпусу чанков")
top_k = st.slider("Количество результатов для поиска", 1, 10, 5)

if query:
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    st.subheader(f"Результаты поиска по запросу: «{query}»")
    for rank, idx in enumerate(indices[0]):
        chunk = df_chunks.iloc[idx]
        st.markdown(f"**#{rank+1} — Документ {chunk['document_id']}, Тип: {chunk['comment']}**")
        st.write(chunk['chunk_text'])
        st.markdown(f"[Источник]({chunk['source_url']})")
        st.markdown("---")

# Поиск по тексту чанков для фильтрации таблицы
search_text = st.text_input("Поиск по тексту чанков (фильтрация таблицы)")
if search_text:
    pattern = re.compile(re.escape(search_text), re.IGNORECASE)
    df_filtered = df_filtered[df_filtered['chunk_text'].str.contains(pattern)]

df_filtered['chunk_len'] = df_filtered['chunk_text'].str.len()

# Визуализация распределения длины чанков
st.header("Распределение длины чанков")
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df_filtered['chunk_len'], bins=30, kde=True, color='cornflowerblue', edgecolor='black', ax=ax)
mean_len = df_filtered['chunk_len'].mean()
ax.axvline(mean_len, color='red', linestyle='--', label=f'Среднее: {mean_len:.1f}')
ax.set_title('Распределение длины чанков (в символах)', fontsize=16)
ax.set_xlabel('Длина чанка (символы)', fontsize=14)
ax.set_ylabel('Количество чанков', fontsize=14)
ax.legend()
st.pyplot(fig)

# Фильтрация "мусорных" фраз
generic_phrases = [
    'отправили фото',
    'пост состоит из фото',
    'отправил фото',
    'фото',
    'изображение',
    'отправлено фото',
]
df_filtered_clean = filter_generic_texts(df_filtered, generic_phrases)

# Облако слов с кешированием
st.header("Облако слов")
text_all = " ".join(df_filtered_clean['chunk_text'].dropna().astype(str))
wordcloud = generate_wordcloud(text_all)
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
st.pyplot(fig_wc)

# Топ-10 частотных слов
st.header("Топ-10 частотных слов")
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['и', 'в', 'на', 'с', 'что', 'по', 'для', 'не', 'как', 'это', 'но', 'из', 'к', 'от', 'у', 'за', 'со'])
    return [t for t in tokens if t not in stopwords and len(t) > 2]

tokens = tokenize(text_all)
counter = Counter(tokens)
most_common = counter.most_common(10)
for word, freq in most_common:
    st.write(f"{word}: {freq}")

# Таблица чанков (интерактивная)
st.header("Таблица чанков (интерактивная)")
df_display = df_filtered_clean[['chunk_uid', 'document_id', 'chunk_id', 'title', 'chunk_text', 'comment', 'year', 'source_url']].copy()
st.dataframe(df_display, use_container_width=True)

st.markdown("### Ссылки на источники:")
for idx, row in df_display.iterrows():
    st.markdown(f"- [{row['chunk_uid']}]({row['source_url']}) — {row['title']}")
