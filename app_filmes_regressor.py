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
    layout="wide",  
    initial_sidebar_state="collapsed" # Sidebar inicia fechada
)

# ======================================================
# 2. ESTILOS (CSS)
# ======================================================
def load_css():
    """Injeta o CSS global (Netflix Style + Sidebar + Gradiente)"""
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

body { 
    /* ATUALIZADO: Fundo com Gradiente */
    background-image: radial-gradient(ellipse at center, #2a080a 0%, #141414 70%);
    background-attachment: fixed;
    background-size: cover;
}
[data-testid="stAppViewContainer"] {
    /* ATUALIZADO: Fundo com Gradiente */
    background-image: radial-gradient(ellipse at center, #2a080a 0%, #141414 70%);
    background-attachment: fixed;
    background-size: cover;
    color: var(--netflix-white);
    
    /* Remove a cor sólida para o gradiente aparecer */
    background-color: transparent; 
}

/* Deixa o header transparente para o gradiente passar */
[data-testid="stHeader"] {
    background-color: transparent;
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
.capa-item { font-size: 1.2rem; color: var(--text-grey); }
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

/* Cards de Resultado */
.movie-card {
    background-color: var(--netflix-dark-grey);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem; 
    border: 2px solid var(--glow-yellow);
    box-shadow: 0 0 15px 3px rgba(245, 213, 10, 0.4);
    transition: all 0.3s ease;
    display: flex; flex-direction: column; height: 100%; min-height: 520px; /* Aumentado para mais campos */
}
.movie-card:hover { transform: translateY(-5px) scale(1.03); box-shadow: 0 0 25px 7px rgba(245, 213, 10, 0.6); }
.movie-card-title { font-size: 1.4rem; font-weight: bold; color: var(--netflix-white); margin-bottom: 1rem; }
.movie-card-content { font-size: 1.1rem; line-height: 1.7; color: var(--text-grey); flex-grow: 1; }
.movie-card-content strong { color: var(--netflix-white); font-size: 1.1rem; }
.movie-card-score { font-size: 1.1rem; font-weight: bold; color: var(--glow-yellow); margin-top: 1rem; text-align: right; }

/* Estilo da Sidebar (do novo código) */
[data-testid="stSidebar"] {
    background-color: var(--netflix-dark-grey);
}
.metric-card {
    background-color: #1f1f1f;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #444;
    text-align: center;
    margin-top: 10px;
}
.metric-title {
    color: #e559f9;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #ffffff;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# Chamada da função de CSS
load_css()

# ======================================================
# 3. GERENCIAMENTO DE ESTADO (Splash Screen)
# ======================================================
if 'app_iniciado' not in st.session_state:
    st.session_state['app_iniciado'] = False

def iniciar_app():
    st.session_state['app_iniciado'] = True

# ======================================================
# 4. CARREGAMENTO DE DADOS (Global)
# ======================================================

try:
    df_movies = pd.read_csv("./dataset_tratado - Filmes.csv")
    model_data = joblib.load("modelo_recomendacao_ContentBased.joblib")
    
    encoder = model_data["encoder"]
    movies_base = model_data["train_data"]
    features = model_data["features"]
except FileNotFoundError as e:
    st.error(f"Erro: Arquivo de dados não encontrado: {e.filename}")
    st.warning("Por favor, verifique se os arquivos 'dataset_tratado - Filmes.csv' e 'modelo_recomendacao_ContentBased.joblib' estão no diretório raiz.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
    st.stop()


# ======================================================
# 5. LÓGICA DE RECOMENDAÇÃO 
# ======================================================

# PESOS DE IMPORTÂNCIA 
feature_weights = {
    "main_actor": 3.0,
    "director": 2.5,
    "genre": 2.0,
    "theme": 1.8,
    "release_decade": 1.5,
    "producer": 1.2,
    "premios": 1.3,
    "age_rating": 1.0,
    "country": 0.8
}

def ajustar_pesos_por_input(user_dict, base_weights):
    adjusted = {}
    for feature, weight in base_weights.items():
        if user_dict[feature] == "Outros":  
            adjusted[feature] = 0  # não selecionado → não influencia
        else:
            adjusted[feature] = weight
    return adjusted

def aplicar_pesos(vetor, encoder, feature_weights):
    vetor = vetor.toarray()
    col_names = encoder.get_feature_names_out()
    for i, name in enumerate(col_names):
        for feature, weight in feature_weights.items():
            if name.startswith(f"cat__{feature}"):
                vetor[:, i] *= weight
    return vetor

def calcular_assertividade(model_data, K=5):
    """Calcula a assertividade média do modelo."""
    # Garante que a matriz de similaridade existe no arquivo
    if "similarity_matrix_test" not in model_data:
        st.sidebar.error("Matriz de similaridade de teste não encontrada no modelo.")
        return 0.0

    similarity_test = model_data["similarity_matrix_test"]
    topk_similarities = []
    
    for i in range(similarity_test.shape[0]):
        sims = similarity_test[i]
        top_k = np.sort(sims)[-K:]  # pega os K mais similares
        topk_similarities.append(np.mean(top_k))
        
    mean_similarity = np.mean(topk_similarities)
    return mean_similarity

def recomendar_filmes(user_dict, top_n=10):
    """Gera recomendações com base no dicionário do usuário."""
     # 1 — Ajusta os pesos com base no input do usuário
    pesos_dinamicos = ajustar_pesos_por_input(user_dict, feature_weights)

    # 2 — Cria vetor do usuário com pesos
    user_df = pd.DataFrame([user_dict])
    user_vector = encoder.transform(user_df[features])
    user_vector = aplicar_pesos(user_vector, encoder, pesos_dinamicos)

    # 3 — Vetor dos filmes com os mesmos pesos
    movies_vectors = encoder.transform(movies_base[features])
    movies_vectors = aplicar_pesos(movies_vectors, encoder, pesos_dinamicos)

    # 4 — Similaridade de cosseno
    sims = cosine_similarity(user_vector, movies_vectors).flatten()

    # 5 — Ranking
    top_idx = np.argsort(sims)[::-1][:top_n]

    return movies_base.iloc[top_idx], sims[top_idx]

# ======================================================
# 6. SIDEBAR 
# ======================================================
score = calcular_assertividade(model_data, K=5)
percent = score * 100

st.sidebar.header("📊 Performance do Modelo")
st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Assertividade Média</div>
        <div class="metric-value">{percent:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("") # Espaço extra
st.sidebar.info("A assertividade mede a similaridade média das recomendações de teste (K=5).")


# ======================================================
# 7. FUNÇÕES DE UI (Interface)
# ======================================================
def exibir_capa(callback_iniciar):
    """Exibe a tela de capa/splash screen."""
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        st.markdown("""
<div class="capa-container">
    <div class="capa-titulo">PROJETO A3</div>
    <div class="capa-subtitulo">Inteligência Artificial (Unifacs)</div>
    <div classa-grid">
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
        st.button("▶ INICIAR APLICAÇÃO", on_click=callback_iniciar, use_container_width=True)

def campo_misto(label, series):
    
    # Converte todos os itens para string ANTES de ordenar
    opcoes_unicas = series.dropna().astype(str).unique()
    opcoes = [""] + sorted(list(opcoes_unicas))
    return st.selectbox(
        label,
        options=opcoes,
        index=0,
        format_func=lambda x: f"Digite ou selecione..." if x == "" else x
    )

def exibir_formulario_inputs(df_movies):
    """Exibe a coluna central com os inputs (layout antigo + inputs novos)."""
    inputs = {}
    col_vazia1, col_inputs, col_vazia2 = st.columns([0.75, 1.5, 0.75])

    with col_inputs:
        st.write("Escolha as características do filme desejado e encontre sua próxima sessão!")

        col1, col2 = st.columns(2)
        with col1: inputs["decade"] = campo_misto("📅 Década do Filme", df_movies["Decada do Filme"])
        with col2: inputs["genre"] = campo_misto("🎭 Gênero", df_movies["Genero"])

        col1, col2 = st.columns(2)
        with col1: inputs["theme"] = campo_misto("💡 Tema", df_movies["Tematica"])
        with col2: inputs["director"] = campo_misto("🎬 Diretor", df_movies["Diretor"])

        col1, col2 = st.columns(2)
        with col1: inputs["actor"] = campo_misto("⭐ Ator/Atriz", df_movies["Estrela"])
        with col2: inputs["country"] = campo_misto("🌍 País de Origem", df_movies["Pais de Origem"])

        col1, col2 = st.columns(2)
        with col1: inputs["producer"] = campo_misto("🏢 Produtora", df_movies["Produtora"])
        with col2: inputs["age"] = campo_misto("🔞 Público Alvo", df_movies["Publico Alvo"])

        col1, col2 = st.columns(2)
        with col1: inputs["release"] = campo_misto("🚀 Tipo de Lançamento", df_movies["Lancamento"])
        with col2: inputs["premios"] = st.selectbox("🏆 Indicado ou ganhador de prêmios?",
                                                    ["Sim", "Não"], index=0)
        
        st.markdown("---", unsafe_allow_html=True) 
        inputs["buscar"] = st.button("🔍 Buscar recomendações")
    
    return inputs

def exibir_resultados(recomendados, scores):
    """Exibe os resultados em cartões (layout antigo + campos novos)."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div id="inicio_resultados" style="position: relative; top: -100px;"></div>', unsafe_allow_html=True)
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
                    <strong>💡 Tema:</strong> {row['theme']}<br>
                    <strong>🎬 Diretor:</strong> {row['director']}<br>
                    <strong>⭐ Ator/Atriz:</strong> {row['main_actor']}<br>
                    <strong>🏢 Produtora:</strong> {row['producer']}<br>
                    <strong>📅 Década:</strong> {row['release_decade']}<br>
                    <strong>🔞 Público:</strong> {row['age_rating']}
                </div>
                <div class="movie-card-score">
                    Similaridade: {scores[idx]:.2f} ✨
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Script de scroll aqui
    components.html("""
        <script>
            setTimeout(function() {
                const element = window.parent.document.getElementById('inicio_resultados');
                if (element) { 
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' }); 
                }
            }, 100); 
        </script>
    """, height=0)

# ======================================================
# 8. EXECUÇÃO PRINCIPAL
# ======================================================

# Se o app não foi iniciado, mostra a capa
if not st.session_state['app_iniciado']:
    exibir_capa(iniciar_app)

# Caso contrário, mostra a aplicação principal
else:
    # --- INTERFACE PRINCIPAL ---
    st.title("🎬 Sistema de Recomendação de Filmes")
    
    # Exibe o formulário e coleta os inputs
    inputs_usuario = exibir_formulario_inputs(df_movies)

    # --- PROCESSAMENTO E RESULTADOS ---
    if inputs_usuario["buscar"]:
        st.balloons()
        
        # Mapeamento 
        user_dict = {
            "genre": inputs_usuario["genre"] or "Outros",
            "theme": inputs_usuario["theme"] or "Outros",
            "main_actor": inputs_usuario["actor"] or "Outros",
            "director": inputs_usuario["director"] or "Outros",
            "producer": inputs_usuario["producer"] or "Outros",
            "country": inputs_usuario["country"] or "Outros",
            "age_rating": inputs_usuario["age"] or "Outros",
            "release_decade": inputs_usuario["decade"] or "Outros",
            "release_type": inputs_usuario["release"] or "Outros",
            "premios": inputs_usuario["premios"] or "Talvez",
        }
        
        # Gera as recomendações
        recomendados, scores = recomendar_filmes(
            user_dict, 
            top_n=9 # Pegando 9 para caber nas 3 colunas
        )
        
        # Exibe os cartões de resultado
        exibir_resultados(recomendados, scores)