import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st 


# PESOS E LÓGICA DE RECOMENDAÇÃO

# Pesos de Importância 
pesos_caracteristicas = {
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
    """Ajusta os pesos, zerando se o usuário escolher 'Outros' (vazio)."""
    adjusted = {}
    for feature, weight in base_weights.items():
        user_value = user_dict.get(feature) 
        
        # Se o valor for "Outros" ou "Talvez" (usado para Não/Sim de prêmios se não for Sim), o peso é zero.
        if user_value is None or user_value == "Outros" or user_value == "Não":  
            adjusted[feature] = 0  
        else:
            adjusted[feature] = weight
            
    return adjusted

def aplicar_pesos(vetor, encoder, feature_weights):
    """Aplica os pesos às colunas do vetor one-hot-encoded."""
    vetor = vetor.toarray()
    col_names = encoder.get_feature_names_out()
    
    for i, name in enumerate(col_names):
        for feature, weight in feature_weights.items():
            # Verifica se o nome da coluna no encoder começa com a feature
            if name.startswith(f"cat__{feature}"):
                vetor[:, i] *= weight
    return vetor

def calcular_assertividade(dados_modelo, K=5):
    """Calcula a assertividade média do modelo a partir da matriz de teste."""
    if "similarity_matrix_test" not in dados_modelo:
        return 0.0

    similarity_test = dados_modelo["similarity_matrix_test"]
    topk_similarities = []
    
    for i in range(similarity_test.shape[0]):
        sims = similarity_test[i]
        top_k = np.sort(sims)[-K:]
        topk_similarities.append(np.mean(top_k))
        
    mean_similarity = np.mean(topk_similarities)
    return mean_similarity

def recomendar_filmes(user_dict, encoder, movies_base, feature_weights, top_n=10):
    """Gera recomendações com base no dicionário do usuário."""
    
    # 1 — Ajusta os pesos
    pesos_dinamicos = ajustar_pesos_por_input(user_dict, feature_weights)

    # 2 — Cria vetor do usuário com pesos
    user_df = pd.DataFrame([user_dict])
    
    # As features para o encoder devem ser as que foram usadas no treinamento
    features_para_encoder = list(feature_weights.keys()) + ["release_type"]
    
    user_vector = encoder.transform(user_df[features_para_encoder])
    user_vector = aplicar_pesos(user_vector, encoder, pesos_dinamicos)

    # 3 — Vetor dos filmes de treino com os mesmos pesos
    movies_vectors = encoder.transform(movies_base[features_para_encoder])
    movies_vectors = aplicar_pesos(movies_vectors, encoder, pesos_dinamicos)

    # 4 — Similaridade de cosseno
    sims = cosine_similarity(user_vector, movies_vectors).flatten()

    # 5 — Ranking
    top_idx = np.argsort(sims)[::-1][:top_n]

    return movies_base.iloc[top_idx], sims[top_idx]