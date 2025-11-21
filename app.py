
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components

# Importando a lógica de recomendação e os pesos do arquivo logica_recomendacao.py
from logica_recomendacao import (
    pesos_caracteristicas, 
    ajustar_pesos_por_input, 
    calcular_assertividade, 
    recomendar_filmes
)

# 1. CONFIGURAÇÃO INICIAL E CARREGAMENTO DE ARQUIVOS

st.set_page_config(
    page_title="Projeto A3 - Recomendação de Filmes",
    page_icon="🍿",
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# CARREGAMENTO DE DADOS (Global)

try:
    df_filmes = pd.read_csv("./dataset_tratado - Filmes.csv")
    dados_modelo = joblib.load("modelo_recomendacao_ContentBased.joblib")
    
    # Objetos essenciais para a recomendação
    encoder = dados_modelo["encoder"]
    base_filmes_treino = dados_modelo["train_data"]
    
except FileNotFoundError as e:
    st.error(f"Erro: Arquivo de dados não encontrado. Verifique 'dataset_tratado - Filmes.csv' e 'modelo_recomendacao_ContentBased.joblib'.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
    st.stop()


# 2. ESTILOS - Carregando do arquivo CSS 

def carregar_arquivo_css(nome_arquivo):
    """Lê e injeta o CSS de um arquivo externo."""
    try:
        with open(nome_arquivo, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo CSS '{nome_arquivo}' não encontrado. Crie o arquivo 'estilos.css'.")

carregar_arquivo_css("estilos.css") 


# 3. GERENCIAMENTO DE ESTADO DA APLICAÇÃO

if 'app_iniciado' not in st.session_state:
    st.session_state['app_iniciado'] = False

def iniciar_app():
    st.session_state['app_iniciado'] = True


# 4. SIDEBAR 

score = calcular_assertividade(dados_modelo, K=5)
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
st.sidebar.write("") 
st.sidebar.info("A assertividade mede a similaridade média das recomendações de teste (K=5).")

# 5. FUNÇÕES DE UI 


def exibir_capa(callback_iniciar):
    """Renderiza a tela de boas-vindas da aplicação antes de iniciar o formulário."""
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        st.markdown("""
<div class="capa-container">
    <div class="capa-titulo">PROJETO A3</div>
    <div class="capa-subtitulo">Inteligência Artificial (Unifacs)</div>
    <div class="capa-grid">
        <div class="capa-item">
            <span class="capa-label">👨‍🏫 Professor</span>
            Adailton de Jesus Cerqueira Junior
        </div>
        <div class="capa-item">
            <span class="capa-label">👥 Integrantes</span>
            Glenda Souza Fernandes dos Santos<br>
            Paulo Henrique Pereira Araujo Piedade<br>
            João Luccas Lordelo Marques<br>
            Marcus Vinicius Lameu Lima<br>        
            Isaac Oliveira Dias<br>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        
    st.write("") 
    b1, b2, b3 = st.columns([1.2, 1, 1.2]) 
    with b2:
        st.button("▶ INICIAR APLICAÇÃO", on_click=callback_iniciar, use_container_width=True)

def campo_misto(label, series, key_name):
    """Cria um selectbox genérico."""
    opcoes_unicas = series.dropna().astype(str).unique()
    opcoes = [""] + sorted(list(opcoes_unicas))
    return st.selectbox(
        label,
        options=opcoes,
        index=0,
        format_func=lambda x: f"Selecione..." if x == "" else x,
        key=key_name
    )

def exibir_formulario_inputs(df_filmes):
    """Exibe o formulário de inputs, com ordem baseada nos pesos."""
    
    # Mapeamento: Chave do Peso 

    input_mapping = {
        "main_actor": ("⭐ Ator/Atriz", "Estrela", "actor"),
        "director": ("🎬 Diretor", "Diretor", "director"),
        "genre": ("🎭 Gênero", "Genero", "genre"),
        "theme": ("💡 Tema", "Tematica", "theme"),
        "release_decade": ("📅 Década do Filme", "Decada do Filme", "decade"),
        "producer": ("🏢 Produtora", "Produtora", "producer"),
        "premios": ("🏆 Indicado/Premiado?", None, "premios"),
        "age_rating": ("🔞 Público Alvo", "Publico Alvo", "age"),
        "country": ("🌍 País de Origem", "Pais de Origem", "country"),
        "release_type_extra": ("🚀 Tipo de Lançamento", "Lancamento", "release"), 
    }
    
    # 1. Ordena as features pelo peso (maior para o menor)
    sorted_features = sorted(
        [k for k in pesos_caracteristicas.keys() if k in input_mapping], 
        key=lambda k: pesos_caracteristicas[k], 
        reverse=True
    )
    
    col_vazia1, col_inputs, col_vazia2 = st.columns([0.75, 1.5, 0.75])

    inputs = {}
    temp_inputs = {}

    with col_inputs:
        st.write("Escolha as características do filme desejado e encontre sua próxima sessão!")

        for i in range(0, len(sorted_features) - 1, 2):
            col1, col2 = st.columns(2)
            
            # --- Input 1 (col1) ---
            feature_key1 = sorted_features[i]
            label1, df_col1, user_dict_key1 = input_mapping[feature_key1]
            
            with col1:
                if feature_key1 == "premios":
                    temp_inputs[feature_key1] = st.selectbox(
                        label1, ["Sim", "Não"], index=0, key=f"input_{feature_key1}"
                    )
                else:
                    temp_inputs[feature_key1] = campo_misto(
                        label1, df_filmes[df_col1], key_name=f"input_{feature_key1}"
                    )

            if i + 1 < len(sorted_features):
                feature_key2 = sorted_features[i + 1]
                label2, df_col2, user_dict_key2 = input_mapping[feature_key2]
                
                with col2:
                    if feature_key2 == "premios":
                         temp_inputs[feature_key2] = st.selectbox(
                            label2, ["Sim", "Não"], index=0, key=f"input_{feature_key2}"
                         )
                    else:
                        temp_inputs[feature_key2] = campo_misto(
                            label2, df_filmes[df_col2], key_name=f"input_{feature_key2}"
                        )
        
        
        col1, col2 = st.columns(2)
        
        feature_key_country = sorted_features[-1]
        label_country, df_col_country, user_dict_key_country = input_mapping[feature_key_country]
        
        with col1:
            temp_inputs[feature_key_country] = campo_misto(
                label_country, df_filmes[df_col_country], key_name=f"input_{feature_key_country}"
            )
        
        label_rel, df_col_rel, user_dict_key_rel = input_mapping["release_type_extra"]
        with col2:
            temp_inputs["release_type_extra"] = campo_misto(
                label_rel, df_filmes[df_col_rel], key_name="input_release_type"
            )


        st.markdown("---", unsafe_allow_html=True) 
        inputs["buscar"] = st.button("🔍 Buscar recomendações")

    for feature_key, (label, df_col, user_dict_key) in input_mapping.items():
        if feature_key == "release_type_extra":
            inputs["release"] = temp_inputs["release_type_extra"]
        else:
            # Encontra a chave de peso correspondente (que está em temp_inputs)
            key_in_temp = [k for k, v in input_mapping.items() if v[2] == user_dict_key and k != "release_type_extra"]
            if key_in_temp:
                inputs[user_dict_key] = temp_inputs[key_in_temp[0]]

    return inputs

def exibir_resultados(recomendados, scores):
    """Exibe os resultados em cartões (HTML/JS embutido)."""
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
            
    # JavaScript para o scroll quando os resultados são exibidos

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



# 6. EXECUÇÃO PRINCIPAL

if not st.session_state['app_iniciado']:
    exibir_capa(iniciar_app)

else:
    st.title("🎬 Sistema de Recomendação de Filmes")
    
    inputs_usuario = exibir_formulario_inputs(df_filmes)

    if inputs_usuario.get("buscar"): 
        st.balloons()
        
        user_dict = {
            "genre": inputs_usuario.get("genre") or "Outros",
            "theme": inputs_usuario.get("theme") or "Outros",
            "main_actor": inputs_usuario.get("actor") or "Outros",
            "director": inputs_usuario.get("director") or "Outros",
            "producer": inputs_usuario.get("producer") or "Outros",
            "country": inputs_usuario.get("country") or "Outros",
            "age_rating": inputs_usuario.get("age") or "Outros",
            "release_decade": inputs_usuario.get("decade") or "Outros",
            "release_type": inputs_usuario.get("release") or "Outros",
            "premios": inputs_usuario.get("premios") or "Talvez",
        }
        
        # Gera as recomendações (chama a função de logica_recomendacao.py)
        recomendados, scores = recomendar_filmes(
            user_dict, encoder, base_filmes_treino, pesos_caracteristicas, 
            top_n=9 
        )
        
        exibir_resultados(recomendados, scores)