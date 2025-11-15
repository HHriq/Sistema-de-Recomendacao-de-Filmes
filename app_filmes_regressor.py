import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit.components.v1 as components

# ======================================================
# 1. CONFIGURAÇÃO INICIAL
# ======================================================
st.set_page_config(
    page_title="Projeto A3 - Recomendação de Filmes",
    page_icon="🍿",
    layout="wide"
)

# ======================================================
# 2. CSS GLOBAL
# ======================================================
st.markdown("""
<style>
:root {
    --netflix-red: #E50914;
    --netflix-black: #141414;
    --netflix-dark-grey: #1f1f1f;
    --netflix-white: #FFFFFF;
    --text-grey: #B3B3B3;
    --glow-yellow: #F5D50A;
}

body { background-color: var(--netflix-black); }
[data-testid="stAppViewContainer"] {
    background-color: var(--netflix-black);
    color: var(--netflix-white);
}

/* --- ESTILOS DA CAPA --- */
.capa-container {
    background: linear-gradient(145deg, #1f1f1f, #141414);
    border: 2px solid var(--netflix-red);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    box-shadow: 0 0 50px rgba(229, 9, 20, 0.25);
    margin-bottom: 30px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

.capa-titulo {
    font-family: 'Arial Black', sans-serif;
    font-size: 3.5rem;
    color: var(--netflix-red);
    text-transform: uppercase;
    margin-bottom: 10px;
    letter-spacing: 2px;
    text-shadow: 2px 2px 4px #000;
}

.capa-subtitulo {
    font-size: 1.4rem;
    color: var(--netflix-white);
    margin-bottom: 30px;
    font-weight: 300;
    border-bottom: 1px solid #333;
    display: inline-block;
    padding-bottom: 10px;
}

.capa-grid {
    display: flex;
    flex-direction: column;
    gap: 20px;
    align-items: center;
    margin-top: 20px;
}

.capa-item {
    font-size: 1.2rem;
    color: var(--text-grey);
}

.capa-label {
    color: var(--netflix-white);
    font-weight: bold;
    font-size: 1.3rem;
    display: block;
    margin-bottom: 5px;
}

/* --- ESTILOS GERAIS --- */
h1 { color: var(--netflix-red) !important; font-weight: bold; text-transform: uppercase; text-align: center; font-size: 2.5rem !important; }
h3 { color: var(--netflix-white) !important; border-bottom: 2px solid var(--netflix-red); padding-bottom: 10px; font-size: 1.75rem !important; margin-bottom: 4rem !important; }

/* Inputs */
[data-testid="stSelectbox"] label { color: var(--text-grey) !important; font-size: 1.1rem !important; font-weight: bold; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background-color: var(--netflix-dark-grey); border-color: #444; color: var(--netflix-white); }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div { color: var(--netflix-white) !important; }
[data-baseweb="popover"] ul { background-color: var(--netflix-dark-grey); border-color: #555; }
[data-baseweb="popover"] ul li:hover { background-color: var(--netflix-red); }

/* Botão */
.stButton>button {
    background-color: var(--netflix-red);
    color: var(--netflix-white);
    border: none;
    border-radius: 30px;
    padding: 0.8rem 2rem;
    font-size: 1.3rem;
    font-weight: bold;
    text-transform: uppercase;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
.stButton>button:hover {
    background-color: #ff1f2b;
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(229, 9, 20, 0.8);
}

/* Cards */
.movie-card {
    background-color: var(--netflix-dark-grey);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem; 
    border: 2px solid var(--glow-yellow);
    box-shadow: 0 0 15px 3px rgba(245, 213, 10, 0.4);
    transition: all 0.3s ease;
    display: flex; flex-direction: column; height: 100%; min-height: 450px; 
}
.movie-card:hover { transform: translateY(-5px) scale(1.03); box-shadow: 0 0 25px 7px rgba(245, 213, 10, 0.6); }
.movie-card-title { font-size: 1.4rem; font-weight: bold; color: var(--netflix-white); margin-bottom: 1rem; }
.movie-card-content { font-size: 1.2rem; line-height: 1.8; color: var(--text-grey); flex-grow: 1; }
.movie-card-content strong { color: var(--netflix-white); font-size: 1.2rem; }
.movie-card-score { font-size: 1.1rem; font-weight: bold; color: var(--glow-yellow); margin-top: 1rem; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. GERENCIAMENTO DE ESTADO
# ======================================================
if 'app_iniciado' not in st.session_state:
    st.session_state['app_iniciado'] = False

def iniciar_app():
    st.session_state['app_iniciado'] = True

# ======================================================
# 4. TELA DE CAPA
# ======================================================
if not st.session_state['app_iniciado']:
    
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        # IMPORTANTE: O HTML está colado na margem esquerda para evitar o erro de indentação
        st.markdown("""
<div class="capa-container">
    <div class="capa-titulo">PROJETO A3</div>
    <div class="capa-subtitulo">Inteligência Artificial (Unifacs)</div>
    <div class="capa-grid">
        <div class="capa-item">
            <span class="capa-label">👨‍🏫 Professor</span>
            Adailton
        </div>
        <div class="capa-item">
            <span class="capa-label">👥 Integrantes</span>
            Glenda • Henrique • Vinicius • Isaac • João
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
    st.write("") 
    
    b1, b2, b3 = st.columns([1.2, 1, 1.2]) 
    
    with b2:
        st.button("▶ INICIAR APLICAÇÃO", on_click=iniciar_app, use_container_width=True)

# ======================================================
# 5. APLICAÇÃO PRINCIPAL
# ======================================================
else:
    # --- FUNÇÕES ---
    def campo_misto(label, series):
        opcoes_unicas = series.dropna().unique().astype(str)
        opcoes_ordenadas = sorted(opcoes_unicas)
        opcoes = [""] + opcoes_ordenadas
        return st.selectbox(label, options=opcoes, index=0, format_func=lambda x: f"Digite ou selecione..." if x == "" else x)

    feature_weights = {
        "main_actor": 3.0, "director": 2.5, "genre": 2.0, "theme": 1.8,
        "release_decade": 1.5, "producer": 1.2, "country": 1.3,
        "age_rating": 1.0, "premios": 0.8
    }

    def aplicar_pesos(vetor, encoder, feature_weights):
        vetor = vetor.toarray()
        col_names = encoder.get_feature_names_out()
        for i, name in enumerate(col_names):
            for feature, weight in feature_weights.items():
                if name.startswith(f"cat__{feature}"):
                    vetor[:, i] *= weight
        return vetor

    try:
        df_movies = pd.read_csv("./dataset_tratado - Filmes.csv")
        model_data = joblib.load("modelo_recomendacao_ContentBased.joblib")
        encoder = model_data["encoder"]
        movies_base = model_data["train_data"]
        features = model_data["features"]
    except FileNotFoundError:
        st.error("Erro: Arquivos de dados não encontrados.")
        st.stop()

    def recomendar_filmes(user_dict, top_n=10):
        user_df = pd.DataFrame([user_dict])
        user_vector = encoder.transform(user_df[features])
        user_vector = aplicar_pesos(user_vector, encoder, feature_weights)
        movies_vectors = encoder.transform(movies_base[features])
        movies_vectors = aplicar_pesos(movies_vectors, encoder, feature_weights)
        sims = cosine_similarity(user_vector, movies_vectors).flatten()
        top_idx = np.argsort(sims)[::-1][:top_n]
        return movies_base.iloc[top_idx], sims[top_idx]

    # --- INTERFACE ---
    st.title(" Sistema de Recomendação de Filmes")

    col_vazia1, col_inputs, col_vazia2 = st.columns([0.75, 1.5, 0.75])

    with col_inputs:
        st.write("Escolha as características do filme desejado e encontre sua próxima sessão!")

        col1, col2 = st.columns(2)
        with col1: input_decade = campo_misto("📅 Década do Filme", df_movies["Decada do Filme"])
        with col2: input_genre = campo_misto("🎭 Gênero", df_movies["Genero"])

        col1, col2 = st.columns(2)
        with col1: input_theme = campo_misto("💡 Tema", df_movies["Tematica"])
        with col2: input_director = campo_misto("🎬 Diretor", df_movies["Diretor"])

        col1, col2 = st.columns(2)
        with col1: input_actor = campo_misto("⭐ Ator/Atriz", df_movies["Estrela"])
        with col2: input_country = campo_misto("🌍 País de Origem", df_movies["Pais de Origem"])

        col1, col2 = st.columns(2)
        with col1: input_producer = campo_misto("🏢 Produtora", df_movies["Produtora"])
        with col2: input_age = campo_misto("🔞 Classificação Indicativa", df_movies["Publico Alvo"])

        col1, col2 = st.columns(2)
        with col1: input_premios = campo_misto("🏆 Indicados a Premios", df_movies["Indicado a Premiações de Cinema"])
        
        st.markdown("---", unsafe_allow_html=True) 
        buscar = st.button("🔍 Buscar recomendações")

    # --- RESULTADOS ---
    if buscar:
        st.balloons()
        user_dict = {
            "genre": input_genre if input_genre else "Outros",
            "theme": input_theme if input_theme else "Outros",
            "main_actor": input_actor if input_actor else "Outros",
            "director": input_director if input_director else "Outros",
            "producer": input_producer if input_producer else "Outros",
            "country": input_country if input_country else "Outros",
            "age_rating": input_age if input_age else "Outros",
            "release_decade": input_decade if input_decade else "Outros",
            "premios": input_premios if input_premios else "0",
            "release_type": "Outros",
        }
        recomendados, scores = recomendar_filmes(user_dict, top_n=9)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div id="inicio_resultados" style="position: absolute; top: -100px; visibility: hidden;"></div>', unsafe_allow_html=True)
        st.subheader("✨ Filmes Recomendados:")
        
        cols = st.columns(3)
        for idx, (i, row) in enumerate(recomendados.iterrows()):
            col = cols[idx % 3]
            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-card-title">🍿 {row['title']}</div>
                    <div class="movie-card-content">
                        <strong>🎭 Gênero:</strong> {row['genre']}<br>
                        <strong>🎬 Diretor:</strong> {row['director']}<br>
                        <strong>⭐ Ator/Atriz:</strong> {row['main_actor']}<br>
                        <strong>🌍 País:</strong> {row['country']}<br>
                        <strong>📅 Década:</strong> {row['release_decade']}
                    </div>
                    <div class="movie-card-score">
                        Similaridade: {scores[idx]:.2f} ✨
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        components.html("""
            <script>
                setTimeout(function() {
                    const element = window.parent.document.getElementById('inicio_resultados');
                    if (element) { element.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                }, 100); 
            </script>
        """, height=0)