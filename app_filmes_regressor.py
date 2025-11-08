# ===============================================
# 🎬 MOVIE RECOMMENDATION SYSTEM (REGRESSION)
# ===============================================

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ===============================================
# 🌗 THEME CONFIGURATION (Dark/Light Mode)
# ===============================================
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for dynamic light/dark styling
st.markdown("""
    <style>
    :root {
        --primary-color: #FF4B4B;
        --accent-color: #FFD166;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Auto-adjust for dark/light mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0E1117;
            --text-color: #FAFAFA;
        }
    }

    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #FFFFFF;
            --text-color: #1C1C1C;
        }
    }

    h1, h2, h3, h4 {
        color: var(--accent-color) !important;
    }

    .stButton>button {
        background: var(--primary-color);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #E53E3E;
        transform: scale(1.05);
    }

    .stRadio label {
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================
# 1. Load dataset
# ===============================================
df_original = pd.read_csv('./Datasets para filmes - A3 - Dataset - A3.csv')
df = df_original.copy()

# ===============================================
# 2. Column mapping
# ===============================================
col_mapping = {
    'Decada do Filme': 'release_decade',
    'Duracao': 'duration',
    'Lancamento': 'release_type',
    'Publico Alvo': 'target_audience',
    'Diretor': 'director',
    'Pais de Origem': 'country',
    'Idioma': 'original_language',
    'Ganhou Oscar': 'oscar_winner',
    'Quantidade de Nomeacoes no Oscar': 'oscar_nominee',
    'Nota': 'rating',
    'Nome do Filme': 'title'
}
df.rename(columns=col_mapping, inplace=True)
df_original.rename(columns=col_mapping, inplace=True)

# ===============================================
# 3. Duration conversion (1h 49m → 109)
# ===============================================
def duration_to_minutes(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?", str(value))
    if match:
        h = int(match.group(1)) if match.group(1) else 0
        m = int(match.group(2)) if match.group(2) else 0
        return h * 60 + m
    return np.nan

df["duration_min"] = df["duration"].apply(duration_to_minutes)

# ===============================================
# 4. Genre and theme split
# ===============================================
df[['main_genre', 'theme']] = df_original['Genero e Tematica'].astype(str).str.split(', ', expand=True).iloc[:, 0:2]
df_original[['main_genre', 'theme']] = df_original['Genero e Tematica'].astype(str).str.split(', ', expand=True).iloc[:, 0:2]

# ===============================================
# 5. Clean relevant columns
# ===============================================
cols = [
    'release_decade', 'duration_min', 'release_type', 'target_audience',
    'director', 'main_genre', 'theme', 'country', 'original_language',
    'oscar_nominee', 'oscar_winner', 'rating', 'title'
]
df = df[cols].dropna()

# ===============================================
# 6. Encode categorical features
# ===============================================
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'title':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ===============================================
# 7. Train/test split + model
# ===============================================
X = df.drop(columns=['title', 'rating'])
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42, max_depth=6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


# ===============================================
# 8. Streamlit UI
# ===============================================
st.title("🎥 Movie Recommender System")
st.caption("A smart, content-based movie recommender powered by Decision Tree Regression 🌳")

st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**R² Score:** {r2:.2f}")
st.sidebar.write(f"**Mean Absolute Error:** {mae:.2f}")

st.markdown("---")
st.subheader("✨ Customize Your Movie Preferences")

# === Helper: unique options ===
def unique_options(col):
    return sorted(df_original[col].dropna().unique().tolist())

# === Add "(Any)" option ===
def add_any_option(lst):
    clean_list = sorted([x for x in lst if pd.notna(x)])
    return ["(Any)"] + clean_list

# === Clean multiple countries/languages ===
df_original['country'] = df_original['country'].astype(str).apply(lambda x: x.split(",")[0].strip())
df_original['original_language'] = df_original['original_language'].astype(str).apply(lambda x: x.split(",")[0].strip())

unique_genres = add_any_option(df_original["main_genre"].dropna().unique().tolist())
unique_themes = add_any_option(df_original["theme"].dropna().unique().tolist())
unique_countries = add_any_option(df_original["country"].dropna().unique().tolist())
unique_languages = add_any_option(df_original["original_language"].dropna().unique().tolist())

# === Form inputs ===
release_decade = st.selectbox("Release Decade", unique_options("release_decade"))

duration_options = {
    "Between 1h and 1h30m": (60, 90),
    "Between 1h30m and 2h": (90, 120),
    "Between 2h and 2h30m": (120, 150),
    "More than 2h30m": (150, 240)
}
st.subheader("⏱️ Movie Length")
duration_choice = st.radio("Select duration range:", list(duration_options.keys()), index=1)
dur_min, dur_max = duration_options[duration_choice]
duration = (dur_min + dur_max) / 2

release_type = st.selectbox("Release Type", unique_options("release_type"))
target_audience = st.selectbox("Target Audience", unique_options("target_audience"))
director = st.selectbox("Director", ["(Any)"] + unique_options("director"))
genre = st.selectbox("🎭 Main Genre", unique_genres)
theme = st.selectbox("🎯 Theme", unique_themes)
country = st.selectbox("🌍 Country of Origin", unique_countries)
language = st.selectbox("🗣️ Original Language", unique_languages)

# === Oscar preferences ===
st.subheader("🏆 Oscar Preferences")
oscar_choices = ["Yes", "No", "Maybe"]
oscar_map = {"Yes": 1.0, "Maybe": 0.5, "No": 0.0}

oscar_nominee_ans = st.radio("Prefer Oscar-nominated movies?", oscar_choices, index=1)
oscar_winner_ans = st.radio("Prefer Oscar-winning movies?", oscar_choices, index=1)

oscar_nominee = oscar_map[oscar_nominee_ans]
oscar_winner = oscar_map[oscar_winner_ans]

# ===============================================
# 9. Generate Recommendations
# ===============================================
if st.button("🎬 Generate Recommendations"):
    user_input = {
        'release_decade': release_decade,
        'duration': duration,
        'release_type': release_type,
        'target_audience': target_audience,
        'director': director if director != "(Any)" else None,
        'main_genre': None if genre == "(Any)" else genre,
        'theme': None if theme == "(Any)" else theme,
        'country': None if country == "(Any)" else country,
        'original_language': None if language == "(Any)" else language,
        'oscar_nominee': oscar_nominee,
        'oscar_winner': oscar_winner
    }

    # Encode input
    encoded_input = []
    for col in X.columns:
        val = user_input.get(col, "")
        if col in label_encoders:
            try:
                code = label_encoders[col].transform([str(val)])[0]
            except ValueError:
                code = -1
        else:
            try:
                code = float(val)
            except ValueError:
                code = 0
        encoded_input.append(code)

    input_array = np.array(encoded_input, dtype=float).reshape(1, -1)

    # Predict rating
    predicted_rating = model.predict(input_array)[0]
    similarities = cosine_similarity(X, input_array).flatten()
    df['similarity'] = similarities
    recommendations = df.sort_values(by='similarity', ascending=False).head(10)

    st.success(f"🎯 Predicted rating for your preferences: **{predicted_rating:.2f}** ⭐")
    st.markdown("### 🍿 Recommended Movies:")
    st.dataframe(
        recommendations[['title', 'rating', 'similarity']]
        .rename(columns={'title': 'Title', 'rating': 'Rating', 'similarity': 'Similarity'})
        .reset_index(drop=True)
    )
