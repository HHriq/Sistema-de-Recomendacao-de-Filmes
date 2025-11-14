import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
# Trocado StandardScaler por MinMaxScaler
from sklearn.preprocessing import MinMaxScaler 

# ----------------------------
# Configuração da página
# ----------------------------
st.set_page_config(
    page_title="🎬 Recomendador Híbrido (Classificação + Similaridade)",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --primary-color: #FF4B4B; --accent-color: #FFD166; }
    [data-testid="stAppViewContainer"] { background-color: var(--background-color); color: var(--text-color); }
    @media (prefers-color-scheme: dark) { :root { --background-color: #0E1117; --text-color: #FAFAFA; } }
    @media (prefers-color-scheme: light) { :root { --background-color: #FFFFFF; --text-color: #1C1C1C; } }
    h1,h2,h3,h4 { color: var(--accent-color) !important; }
    .stButton>button { background: var(--primary-color); color: white; border-radius:12px; padding:0.5em 1em; }
    .stButton>button:hover { transform: scale(1.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers de preprocessamento
# ----------------------------
def duration_to_minutes(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?", str(value))
    if match:
        h = int(match.group(1)) if match.group(1) else 0
        m = int(match.group(2)) if match.group(2) else 0
        return h * 60 + m
    try:
        return float(value)
    except Exception:
        return np.nan

def split_multi_tokens(cell):
    if pd.isna(cell):
        return []
    parts = [p.strip() for p in str(cell).split(",") if p.strip()!=""]
    return parts

@st.cache_data
def get_all_tokens(df_column):
    if df_column is None:
        return []
    tokens_series = df_column.fillna("").apply(split_multi_tokens)
    all_tokens = [t for sub in tokens_series for t in sub]
    token_counts = Counter(all_tokens)
    tokens = sorted([t for t, c in token_counts.items() if t != ""])
    return tokens

def build_multihot(df, column):
    """Retorna DataFrame multihot (separador '::')"""
    tokens_series = df[column].fillna("").apply(split_multi_tokens)
    all_tokens = [t for sub in tokens_series for t in sub]
    token_counts = Counter(all_tokens)
    tokens = sorted([t for t, c in token_counts.items() if t != ""])
    if not tokens:
        return pd.DataFrame(index=df.index)
    mh = pd.DataFrame(0, index=df.index, columns=[f"{column}::{t}" for t in tokens])
    for i, toks in enumerate(tokens_series):
        for t in toks:
            mh.at[i, f"{column}::{t}"] = 1
    return mh

def safe_get_dummies(df, col, prefix=None):
    """Retorna DataFrame one-hot (separador '_' por padrão)"""
    if col not in df.columns:
        return pd.DataFrame(index=df.index)
    s = df[col].fillna("__(missing)__").astype(str)
    return pd.get_dummies(s, prefix=prefix or col)

# ----------------------------
# Load & preprocess
# ----------------------------
@st.cache_data
def load_and_preprocess(path="./dataset_tratado.csv"):
    df0 = pd.read_csv(path)
    df_original = df0.copy()

    col_map = {
        "id": "id", "Nome do Filme": "title", "Ano de Lancamento": "year",
        "Decada do Filme": "release_decade", "Duracao": "duration", "Lancamento": "release_type",
        "Classificao do Filme": "classification", "Publico Alvo": "target_audience",
        "Diretor": "director", "Estrela": "star", "Genero e Tematica": "genre_theme",
        "Pais de Origem": "country", "Produtora": "production_company", "Idioma": "original_language",
        "Ganhou Oscar": "oscar_winner", "Quantidade de Nomeacoes - Totais": "n_nom_total",
        "Quantidade de Nomeacoes no Oscar": "n_nom_oscar", "Nota": "rating",
    }
    df0 = df0.rename(columns={k: v for k, v in col_map.items() if k in df0.columns})
    df = df0.copy()

    if "duration" in df.columns:
        df["duration_min"] = df["duration"].apply(duration_to_minutes)
    else:
        df["duration_min"] = np.nan
    if "genre_theme" in df.columns:
        df["genre_theme"] = df["genre_theme"].astype(str)
    else:
        df["genre_theme"] = ""
    if "country" in df.columns:
        df["country_primary"] = df["country"].astype(str).apply(lambda x: str(x).split(",")[0].strip())
    else:
        df["country_primary"] = "__(missing)__"
    if "original_language" in df.columns:
        df["original_language_primary"] = df["original_language"].astype(str).apply(lambda x: str(x).split(",")[0].strip())
    else:
        df["original_language_primary"] = "__(missing)__"
        
    for num in ["n_nom_total", "n_nom_oscar", "rating", "duration_min", "year"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")
        else:
            df[num] = np.nan
            
    return df, df_original

# ----------------------------
# Construir matriz de características
# ----------------------------
@st.cache_data
def build_feature_matrix(df):
    numeric_cols = []
    for c in ["duration_min", "year", "n_nom_total", "n_nom_oscar"]:
        if c in df.columns:
            numeric_cols.append(c)
    num_df = df[numeric_cols].fillna(0).astype(float)
    
    # Gênero usa '::' como separador
    genre_mh = build_multihot(df, "genre_theme")
    
    # O resto usa '_' como separador (padrão do pd.get_dummies)
    oh_release_decade = safe_get_dummies(df, "release_decade", prefix="decade")
    oh_release_type = safe_get_dummies(df, "release_type", prefix="release")
    oh_target = safe_get_dummies(df, "target_audience", prefix="target")
    oh_director = safe_get_dummies(df, "director", prefix="director")
    oh_star = safe_get_dummies(df, "star", prefix="star")
    oh_country = safe_get_dummies(df, "country_primary", prefix="country")
    oh_language = safe_get_dummies(df, "original_language_primary", prefix="lang")
    oh_oscar_winner = safe_get_dummies(df, "oscar_winner", prefix="oscar_winner")

    mats = [
        num_df, genre_mh, oh_release_decade, oh_release_type, oh_target,
        oh_director, oh_star, oh_country, oh_language, oh_oscar_winner,
    ]
    X = pd.concat(mats, axis=1).fillna(0)
    X.columns = [str(c) for c in X.columns]
    return X

# ----------------------------
# Treinar classificador
# ----------------------------
@st.cache_resource
def train_classifier(X, df, rating_threshold=7.0):
    y = (df["rating"].fillna(0) >= rating_threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    proba_all = clf.predict_proba(X)[:, 1]
    return clf, acc, f1, proba_all

# ----------------------------
# Construir vetor do usuário (VERSÃO FINAL CORRIGIDA)
# ----------------------------
def build_user_vector(user_inputs, X_columns, numeric_means):
    """
    Constrói o vetor do usuário de forma "hermética", garantindo que
    cada input afete apenas as colunas corretas com os SEPARADORES CORRETOS.
    """
    v = pd.Series(0.0, index=X_columns, dtype=float)

    # 1. CAMPOS NUMÉRICOS
    # Trata (Any) usando a média
    numeric_fields_keys = {
        "duration_min": "duration_min",
        "year": "year",
        "n_nom_total": "n_nom_total",
        "n_nom_oscar": "n_nom_oscar"
    }
    for key, col_name in numeric_fields_keys.items():
        if col_name in X_columns:
            val = user_inputs.get(key)
            if val not in (None, "(Any)", ""):
                try:
                    v[col_name] = float(val)
                except Exception:
                    v[col_name] = numeric_means[col_name] # fallback
            else:
                v[col_name] = numeric_means[col_name]

    # 2. CAMPOS MULTIHOT (GÊNERO) - Usa '::'
    genres = split_multi_tokens(user_inputs.get("genre_theme", ""))
    for g in genres:
        col = f"genre_theme::{g}" # ex: genre_theme::Animação
        if col in v.index:
            v[col] = 1.0

    # 3. CAMPOS ONE-HOT (CATEGÓRICOS) - Usa '_'
    
    # (prefixo_real_usado_no_X, chave_do_user_inputs, separador)
    # Esta lista agora está correta, combinando o prefixo da matriz X
    # com a chave dos inputs do usuário
    categorical_fields = [
        ("decade", "release_decade", "_"),
        ("release", "release_type", "_"),
        ("target", "target_audience", "_"),
        ("director", "director", "_"),
        ("star", "star", "_"),
        # CORREÇÃO: A chave do user_inputs era 'country' e 'original_language'
        # e o prefixo da matriz era 'country' e 'lang'
        ("country", "country", "_"), 
        ("lang", "original_language", "_"), 
        ("oscar_winner", "oscar_winner", "_"),
    ]


    for prefix, key, sep in categorical_fields:
        val = user_inputs.get(key)
        
        if val not in (None, "(Any)", ""):
            # Constrói o nome da coluna exatamente como o pd.get_dummies()
            # ex: prefix='director', sep='_', val='Andrew Adamson, Vicky Jenson'
            # -> 'director_Andrew Adamson, Vicky Jenson'
            expected_col = f"{prefix}{sep}{val}"
            
            if expected_col in v.index:
                v[expected_col] = 1.0
            else:
                # Fallback (ex: se o valor for "2000" mas a coluna for "decade_2000.0")
                candidates = [c for c in v.index if c.startswith(f"{prefix}{sep}") and str(val) in c]
                for c in candidates:
                    v[c] = 1.0
                    
    return v.values.reshape(1, -1)

# ----------------------------
# MAIN
# ----------------------------
st.title("🎥 Recomendador Híbrido — RandomForestClassifier + Similaridade")
st.caption("Combina probabilidade do classificador com similaridade do cosseno para ranking Top-10.")

# load
try:
    df, df_display = load_and_preprocess()
except FileNotFoundError:
    st.error("Erro: Arquivo 'Datasets para filmes - A3 - Dataset - A3.csv' não encontrado.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os dados: {e}")
    st.stop()
    
X = build_feature_matrix(df)

# Pegamos as médias das colunas numéricas para tratar 'Any'
numeric_cols = [c for c in ["duration_min", "year", "n_nom_total", "n_nom_oscar"] if c in X.columns]
numeric_means = X[numeric_cols].mean().fillna(0)

# train
clf, acc, f1, proba_all = train_classifier(X, df, rating_threshold=7.0)

# sidebar
st.sidebar.header("📊 Desempenho do Classificador")
st.sidebar.write(f"**Accuracy:** {acc:.3f}")
st.sidebar.write(f"**F1-score:** {f1:.3f}")

# user inputs (form)
st.markdown("---")
st.subheader("✨ Personalize suas preferências")

# helper lists for selects
def unique_list(col):
    if col in df_display.columns:
        vals = df_display[col].dropna().unique().tolist()
        vals = sorted([str(v) for v in vals])
        return ["(Any)"] + vals
    return ["(Any)"]

# 1. Obter listas de opções
decade_opts = unique_list("Decada do Filme")
release_type_opts = unique_list("Lancamento")
target_opts = unique_list("Publico Alvo")
director_opts = unique_list("Diretor")
star_opts = unique_list("Estrela")
country_opts = unique_list("Pais de Origem")
lang_opts = unique_list("Idioma")
genre_opts_column = "Genero e Tematica" if "Genero e Tematica" in df_display.columns else None
genre_opts = get_all_tokens(df_display[genre_opts_column]) if genre_opts_column else []


# 3. Criar layout do formulário
col1, col2 = st.columns(2)
with col1:
    sel_decade = st.selectbox("Década de Lançamento", decade_opts)

    if sel_decade == "(Any)":
        df_for_years = df_display
    else:
        df_for_years = df_display[df_display["Decada do Filme"].astype(str) == sel_decade]
    
    year_opts = ["(Any)"]
    if "Ano de Lancamento" in df_for_years.columns:
         years_clean = pd.to_numeric(df_for_years["Ano de Lancamento"], errors='coerce').dropna()
         year_opts += sorted(years_clean.astype(int).astype(str).unique().tolist())

    sel_year = st.selectbox("Ano de Lançamento", year_opts)
    
    # Slider de Duração em Horas
    duration_choice_hours = st.slider("Duração média (horas)", 0.5, 5.0, 1.7, step=0.1)
    
    sel_release_type = st.selectbox("Tipo de Lançamento", release_type_opts)
    sel_target = st.selectbox("Público Alvo", target_opts)

with col2:
    sel_director = st.selectbox("Diretor", director_opts)
    sel_star = st.selectbox("Estrela", star_opts)
    sel_country = st.selectbox("País de Origem", country_opts)
    sel_lang = st.selectbox("Idioma Original", lang_opts)
    sel_genre_theme_list = st.multiselect("Gêneros / Temáticas", genre_opts)
    sel_genre_theme = ", ".join(sel_genre_theme_list)

# Oscar preference
st.subheader("🏆 Preferências de Oscar")
oscar_choice = st.selectbox("Preferência por filmes vencedores do Oscar?", ["(Any)", "Yes", "No"])
min_rating_for_display = st.slider("Nota mínima para aparecer no Top 10 (apenas filtro de exibição)", 0.0, 10.0, 0.0)

# combine inputs
user_inputs = {
    "release_decade": None if sel_decade == "(Any)" else sel_decade,
    "year": None if sel_year == "(Any)" else sel_year,
    "duration_min": duration_choice_hours * 60, # Converte horas de volta para minutos
    "release_type": None if sel_release_type == "(Any)" else sel_release_type,
    "target_audience": None if sel_target == "(Any)" else sel_target,
    "director": None if sel_director == "(Any)" else sel_director,
    "star": None if sel_star == "(Any)" else sel_star,
    "country": None if sel_country == "(Any)" else sel_country,
    "original_language": None if sel_lang == "(Any)" else sel_lang,
    "genre_theme": sel_genre_theme,
    "oscar_winner": None if oscar_choice == "(Any)" else oscar_choice,
    # (Any) para nominação será tratado pelo build_user_vector
    "n_nom_total": None, 
    "n_nom_oscar": None,
}

st.markdown("---")

# Botão de gerar recomendações
if st.button("🎬 Gerar Top 10 Recomendados"):
    with st.spinner("Calculando recomendações..."):
        
        # Passa as médias para o builder tratar o (Any)
        user_vec = build_user_vector(user_inputs, X.columns, numeric_means)

        # Usa MinMaxScaler
        scaler = MinMaxScaler()
        
        X_scaled = X.copy()
        if numeric_cols:
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            user_vec_df = pd.DataFrame(user_vec, columns=X.columns)
            user_vec_df[numeric_cols] = scaler.transform(user_vec_df[numeric_cols])
            user_vec = user_vec_df.values

        # 1) Probabilidade
        proba = clf.predict_proba(X)[:, 1]

        # 2) Similaridade
        X_for_sim = X_scaled.values.astype(float)
        X_for_sim = np.nan_to_num(X_for_sim, nan=0.0)
        sims = cosine_similarity(X_for_sim, user_vec.astype(float)).flatten()

        # 3) Normalizar
        def normalize_arr(a):
            a = np.array(a, dtype=float)
            if a.max() - a.min() <= 1e-9:
                return np.zeros_like(a)
            return (a - a.min()) / (a.max() - a.min())

        proba_n = normalize_arr(proba)
        sims_n = normalize_arr(sims)

        # combinar score
        alpha = 0.65
        beta = 1.0 - alpha
        final_score = alpha * proba_n + beta * sims_n

        # montar resultado
        result_df = df.copy().reset_index(drop=True)
        result_df["match_proba"] = proba
        result_df["similarity"] = sims
        result_df["score"] = final_score
        if "rating" in result_df.columns and min_rating_for_display > 0:
            result_df = result_df[result_df["rating"].fillna(0) >= min_rating_for_display]

        topk = result_df.sort_values(by="score", ascending=False).head(10)

    # Exibir resultado
    st.success("✅ Recomendação gerada!")
    st.markdown("### 🍿 Top 10 filmes recomendados")
    cols_show = ["title", "year", "rating", "match_proba", "similarity", "score"]
    cols_present = [c for c in cols_show if c in topk.columns]
    st.dataframe(topk[cols_present].rename(columns={
        "title": "Título",
        "year": "Ano",
        "rating": "Nota",
        "match_proba": "Probabilidade (classificador)",
        "similarity": "Similaridade (cosseno)",
        "score": "Score combinado"
    }).reset_index(drop=True))

    st.markdown(
        """
        **Como o ranking foi calculado:**  
        - `Probabilidade (classificador)`: probabilidade do RandomForestClassifier do filme ser 'relevante' (treinado usando nota >= 7 como proxy).  
        - `Similaridade (cosseno)`: quão parecido o filme é com o vetor de preferências informado.  
        - `Score combinado`: alpha * probabilidade + (1-alpha) * similaridade (alpha = 0.65 por padrão).
        """
    )

# Footer
st.markdown("---")
st.caption("Modelo híbrido: RandomForestClassifier (probabilidade) + Similaridade do Cosseno (vetor do usuário). Ajuste o parâmetro alpha no código para dar mais/menos peso ao classificador.")