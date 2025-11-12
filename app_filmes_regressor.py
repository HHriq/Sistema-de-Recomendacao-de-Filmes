import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


st.set_page_config(
    page_title="🎬 Sistema de Recomendação de Filmes",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    :root {
        --primary-color: #FF4B4B;
        --accent-color: #FFD166;
    }
    [data-testid="stAppViewContainer"] { background-color: var(--background-color); color: var(--text-color); }
    @media (prefers-color-scheme: dark) { :root { --background-color: #0E1117; --text-color: #FAFAFA; } }
    @media (prefers-color-scheme: light) { :root { --background-color: #FFFFFF; --text-color: #1C1C1C; } }
    h1, h2, h3, h4 { color: var(--accent-color) !important; }
    .stButton>button { background: var(--primary-color); color: white; border-radius: 12px; padding: 0.6em 1.2em; border: none; transition: all 0.3s ease; }
    .stButton>button:hover { background: #E53E3E; transform: scale(1.05); }
    .stRadio label { color: var(--text-color); }
    </style>
""", unsafe_allow_html=True)


# ===============================================
# 1. FUNÇÃO CACHEADA PARA CARREGAR E PREPARAR DADOS 
# ===============================================
@st.cache_data
def load_data(csv_path):
    # Passo 1: Carregar
    df_original = pd.read_csv(csv_path)
    df = df_original.copy()

    # Passo 2: Mapeamento
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
        'Nome do Filme': 'title',
        'Produtora': 'production_company'
    }
    df.rename(columns=col_mapping, inplace=True)
    df_original.rename(columns=col_mapping, inplace=True)

    # Passo 3: Conversão de Duração
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

    # Passo 4: Divisão de Gênero
    df[['main_genre', 'theme']] = df_original['Genero e Tematica'].astype(str).str.split(', ', expand=True).iloc[:, 0:2]
    df_original[['main_genre', 'theme']] = df_original['Genero e Tematica'].astype(str).str.split(', ', expand=True).iloc[:, 0:2]

    # Passo 5: Limpar Colunas 
    cols = [
        'release_decade', 'duration_min', 'release_type', 'target_audience',
        'director', 'main_genre', 'theme', 'original_language', 
        'oscar_nominee', 'oscar_winner', 'rating', 'title' 
    ]
    df = df[cols].dropna()

    # Limpar dados do df_original que será usado na UI - precisamos mudar isso aqui 
    df_original['original_language'] = df_original['original_language'].astype(str).apply(lambda x: x.split(",")[0].strip())
    
    return df, df_original

# ===============================================
# 2. FUNÇÃO CACHEADA PARA TREINAR O MODELO 
# ===============================================


@st.cache_resource
def train_model(df_clean):
    # Passo 6: One-Hot Encoding 
    colunas_categoricas = [
        'release_decade', 'release_type', 'target_audience', 'director', 
        'main_genre', 'theme', 'original_language' 
    ]
    df_encoded = pd.get_dummies(df_clean, columns=colunas_categoricas, drop_first=True)

    # Passo 7: Split e Treino
    X = df_encoded.drop(columns=['title', 'rating']) 
    y = df_encoded['rating'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        random_state=42, 
        n_estimators=500, 
        max_depth=50,      
        n_jobs=-1         
    )
    model.fit(X_train, y_train)

    # Calcular métricas
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, X, df_encoded, r2, mae

# ===============================================
# 3. CORPO PRINCIPAL
# ===============================================

# Chama as funções cacheadas 
df_clean, df_original_ui = load_data('./Datasets para filmes - A3 - Dataset - A3.csv')
model, X, df_encoded, r2, mae = train_model(df_clean)

# ===============================================
# 8. UI com Streamlit - ainda em desenvolvimento
# ===============================================
st.title("🎥 Sistema de Recomendação de Filmes")
st.caption("Um recomendador de filmes inteligente, baseado em conteúdo, com Regressão 🌳")

st.sidebar.header("📊 Performance do Modelo")
st.sidebar.write(f"**Score R²:** {r2:.2f}")
st.sidebar.write(f"**Erro Absoluto Médio:** {mae:.2f}")

st.markdown("---")
st.subheader("✨ Personalize Suas Preferências de Filmes")

# === Auxiliar: opções únicas ===
def unique_options(col):
    return sorted(df_original_ui[col].dropna().unique().tolist())

# === Adicionar opção "(Qualquer)" ===
def add_any_option(lst):
    clean_list = sorted([x for x in lst if pd.notna(x)])
    return ["(Qualquer)"] + clean_list

# Gera as listas de opções únicas
unique_genres = add_any_option(df_original_ui["main_genre"].dropna().unique().tolist())
unique_themes = add_any_option(df_original_ui["theme"].dropna().unique().tolist())
unique_languages = add_any_option(df_original_ui["original_language"].dropna().unique().tolist())

# === Entradas do formulário ===
release_decade = st.selectbox("Década de Lançamento", unique_options("release_decade"))

duration_options = {
    "Entre 1h e 1h30m": (60, 90),
    "Entre 1h30m e 2h": (90, 120),
    "Entre 2h e 2h30m": (120, 150),
    "Mais de 2h30m": (150, 240)
}
st.subheader("⏱️ Duração do Filme")
duration_choice = st.radio("Selecione o intervalo de duração:", list(duration_options.keys()), index=1)
dur_min, dur_max = duration_options[duration_choice]
duration = (dur_min + dur_max) / 2

release_type = st.selectbox("Tipo de Lançamento", unique_options("release_type"))
target_audience = st.selectbox("Público Alvo", unique_options("target_audience"))
# Este selectbox funciona como uma busca com auto-complete - ainda estamos desenvolvendo pra vê se fica melhor...
director = st.selectbox("Diretor (digite para buscar)", ["(Qualquer)"] + unique_options("director"))
genre = st.selectbox("🎭 Gênero Principal", unique_genres)
theme = st.selectbox("🎯 Temática", unique_themes)
language = st.selectbox("🗣️ Idioma Original", unique_languages)

# === Preferências de Oscar ===
st.subheader("🏆 Preferências de Oscar")
oscar_choices = ["Sim", "Não", "Talvez"]
oscar_map = {"Sim": 1.0, "Talvez": 0.5, "Não": 0.0}

oscar_nominee_ans = st.radio("Prefere filmes indicados ao Oscar?", oscar_choices, index=1)
oscar_winner_ans = st.radio("Prefere filmes vencedores do Oscar?", oscar_choices, index=1)

oscar_nominee = oscar_map[oscar_nominee_ans]
oscar_winner = oscar_map[oscar_winner_ans]

# ===============================================
# 9. Gerar Recomendações
# ===============================================
if st.button("🎬 Gerar Recomendações"):
    
    # Dicionário com as escolhas do usuário
    user_input = {
        'release_decade': release_decade,
        'duration_min': duration,
        'release_type': release_type,
        'target_audience': target_audience,
        'director': director if director != "(Qualquer)" else None,
        'main_genre': None if genre == "(Qualquer)" else genre,
        'theme': None if theme == "(Qualquer)" else theme,
        'original_language': None if language == "(Qualquer)" else language,
        'oscar_nominee': oscar_nominee,
        'oscar_winner': oscar_winner
    }

    # --- LÓGICA DE ENTRADA ONE-HOT ---
    input_series = pd.Series(0, index=X.columns)
    
    input_series['duration_min'] = user_input.get('duration_min', 0)
    input_series['oscar_nominee'] = user_input.get('oscar_nominee', 0.0)
    input_series['oscar_winner'] = user_input.get('oscar_winner', 0.0)
    
    mapa_inputs = {
        'release_decade': user_input['release_decade'],
        'release_type': user_input['release_type'],
        'target_audience': user_input['target_audience'],
        'director': user_input['director'],
        'main_genre': user_input['main_genre'],
        'theme': user_input['theme'],
        'original_language': user_input['original_language'],
    }

    for prefix, value in mapa_inputs.items():
        if value is not None:
            col_name = f"{prefix}_{value}"
            if col_name in input_series.index:
                input_series[col_name] = 1
                
    input_array = input_series.values.reshape(1, -1)

    # Usa o modelo e os dados cacheados
    predicted_rating = model.predict(input_array)[0]
    similarities = cosine_similarity(X, input_array).flatten()
    
    df_encoded['similarity'] = similarities
    recommendations = df_encoded.sort_values(by='similarity', ascending=False).head(10)

    st.success(f"🎯 Nota prevista para suas preferências: **{predicted_rating:.2f}** ⭐")
    st.markdown("### 🍿 Filmes Recomendados:")
    
    st.dataframe(
        recommendations[['title', 'rating', 'similarity']]
        .rename(columns={'title': 'Título', 'rating': 'Nota', 'similarity': 'Similaridade'})
        .reset_index(drop=True)
    )