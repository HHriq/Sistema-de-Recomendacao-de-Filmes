import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import textwrap

MODEL_PATH = './modelo_recomendacao_ContentBased.joblib'

try:
    print("Carregando modelo e dados...")
    model_data = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("ERRO: Arquivo de modelo não encontrado. Execute o treinamento primeiro.")
    sys.exit()

train_df = model_data['train_data']
encoder = model_data['encoder']
features = model_data['features']
NEW_SCORE_COLUMN = 'rating'
print("Transformando dados de treino para similaridade...")
X_train = encoder.transform(train_df[features]) 

fill_values = {
    "genre": "Outros", "theme": "Outros", "main_actor": "Outros", 
    "director": "Outros", "producer": "Outros", "country": "Outros", 
    "age_rating": "Outros", "release_decade": "Outros", 
    "release_type": "Outros", "premios": "Não" 
}
n_recommendations = 10

synthetic_profiles = [
    {"genre": "Thriller Psicológico", "theme": "Suspense / Mistério", "main_actor": "Charlize Theron", "director": "Luca Guadagnino", "producer": "Magnolia Pictures","country": "Estados Unidos", "age_rating": "Família", "release_decade": "2010", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Drama", "theme": "Comédia", "main_actor": "Michael Keaton", "director": "Bruno Barreto", "producer": "Dimension Films","country": "Coreia do Sul", "age_rating": "Jovens", "release_decade": "2020", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Crime", "theme": "Aventura", "main_actor": "Dakota Johnson", "director": "Quentin Tarantino", "producer": "Columbia Pictures","country": "Estados Unidos", "age_rating": "Família", "release_decade": "2000", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Drama", "theme": "Comédia", "main_actor": "Will Smith", "director": "Clint Eastwood", "producer": "Columbia Pictures","country": "Estados Unidos", "age_rating": "Família", "release_decade": "1990", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Crime", "theme": "Aventura", "main_actor": "Isabela Merced", "director": "James Cameron", "producer": "Columbia Pictures","country": "Estados Unidos", "age_rating": "Família", "release_decade": "2020", "release_type":"Cinema", "premios":""},
    {"genre": "Crime", "theme": "Romance", "main_actor": "Ben Affleck", "director": "Sofia Coppola", "producer": "Paramount Pictures","country": "Brasil", "age_rating": "Família", "release_decade": "1990", "release_type":"", "premios":"Sim"},
    {"genre": "Épico de Aventura", "theme": "Romance", "main_actor": "Arnold Schwarzenegger", "director": "Steven Soderbergh", "producer": "Paramount Pictures","country": "Coreia do Sul", "age_rating": "Jovens", "release_decade": "2000", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Drama", "theme": "Comédia", "main_actor": "Robert Downey Jr", "director": "Spike Lee", "producer": "Paramount Pictures","country": "Estados Unidos", "age_rating": "Jovens", "release_decade": "1990", "release_type":"Televisao", "premios":"Nao"},
    {"genre": "Épico de Aventura", "theme": "Crime de Drogas", "main_actor": "Hugh Grant", "director": "Chris Weitz", "producer": "Warner Bros","country": "Estados Unidos", "age_rating": "Jovens", "release_decade": "2020", "release_type":"Televisao", "premios":"Nao"},
    {"genre": "Drama Psicológico", "theme": "Romance Quente / Sensual", "main_actor": "Ryan Reynolds", "director": "Robert Clouse", "producer": "Warner Bros","country": "Itália", "age_rating": "Adultos", "release_decade": "1980", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Drama Psicológico", "theme": "Cinema", "main_actor": "Robert De Niro", "director": "Antonio Margheriti", "producer": "Paramount Pictures","country": "Coreia do Sul", "age_rating": "Adultos", "release_decade": "1980", "release_type":"", "premios":"Nao"},
    {"genre": "Drama", "theme": "Romance Quente / Sensual", "main_actor": "Leonardo DiCaprio", "director": "JJ Abrams", "producer": "Paramount Pictures","country": "Brasil", "age_rating": "Adultos", "release_decade": "2000", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Comédia Romântica", "theme": "Fantasia", "main_actor": "Jamie Foxx", "director": "Elizabeth Chai Vasarhelyi", "producer": "Walt Disney Pictures","country": "Brasil", "age_rating": "Adultos", "release_decade": "2020", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Thriller", "theme": "Fantasia", "main_actor": "Megan Fox", "director": "Tim Burton", "producer": "Harpo Films","country": "Brasil", "age_rating": "Jovens", "release_decade": "1980", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Thriller", "theme": "Mistério", "main_actor": "Emily Blunt", "director": "George Miller", "producer": "Walt Disney Pictures","country": "Coreia do Sul", "age_rating": "Jovens", "release_decade": "2000", "release_type":"", "premios":"Nao"},
    {"genre": "Comédia Romântica", "theme": "Mistério", "main_actor": "Zac Efron", "director": "Tom Holland", "producer": "Harpo Films","country": "Itália", "age_rating": "Jovens", "release_decade": "1990", "release_type":"Cinema", "premios":"Sim"},
    {"genre": "Thriller", "theme": "Assalto", "main_actor": "Robin Williams", "director": "James Ivory", "producer": "TriStar Pictures","country": "França", "age_rating": "Jovens", "release_decade": "2020", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Romance Leve", "theme": "História", "main_actor": "Sandra Bullock", "director": "Brad Bird", "producer": "TriStar Pictures","country": "França", "age_rating": "Família", "release_decade": "1980", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Comédia Vulgar", "theme": "Ficção Científica", "main_actor": "Charlie Sheen", "director": "Zack Snyder", "producer": "Disney Television Animation","country": "Espanha", "age_rating": "Jovens", "release_decade": "2010", "release_type":"Cinema", "premios":"Nao"},
    {"genre": "Comédia Vulgar", "theme": "Ficção Científica", "main_actor": "Meryl Streep", "director": "Sam Raimi", "producer": "Disney Television Animation","country": "França", "age_rating": "Jovens", "release_decade": "2020", "release_type":"Cinema", "premios":"Nao"}
]

def jaccard_coherence(profile_features_dict, recommended_film_row, feature_columns):
    forced_features = set()
    for key, val in profile_features_dict.items():
        if val != 'Outros':
            if key == 'premios':
                if val == 'Sim' or val == 'Não':
                    forced_features.add(val)
            else:
                forced_features.add(val)

    film_features = set()
    for col in feature_columns:
        feature_value = recommended_film_row[col]
        if feature_value != "Outros":
            film_features.add(feature_value)
    
    intersection = forced_features.intersection(film_features)
    union = forced_features.union(film_features)
    
    if not union:
        return 0.0
        
    return len(intersection) / len(union)

all_jaccard_scores = []
detailed_results = []
summary_table_data = []
n_profiles = len(synthetic_profiles)

print("Iniciando Avaliação de Coerência Jaccard...")

for idx_profile, profile_dict in enumerate(synthetic_profiles):
    
    df_profile = pd.DataFrame([profile_dict], columns=features)
    X_profile = encoder.transform(df_profile[features])
    sim_scores = cosine_similarity(X_profile, X_train)[0]
    
    sim_scores_indexed = list(enumerate(sim_scores))
    sim_scores_indexed = sorted(sim_scores_indexed, key=lambda x: x[1], reverse=True)
    top_k_indices = [x[0] for x in sim_scores_indexed[:n_recommendations]]
    
    jaccard_scores_profile = []
    
    current_profile_details = {
        'ID': idx_profile + 1,
        'Perfil Forçado': {k: v for k, v in profile_dict.items() if v != 'Outros' and v != 'Não'},
        'Recomendações': []
    }

    for idx_rank, idx_movie in enumerate(top_k_indices):
        recommended_film_row = train_df.iloc[idx_movie]
        
        j_score = jaccard_coherence(profile_dict, recommended_film_row, features)
        jaccard_scores_profile.append(j_score)
        all_jaccard_scores.append(j_score)

        features_do_filme = {f: recommended_film_row[f] for f in features if recommended_film_row[f] != 'Outros' and recommended_film_row[f] != 'Não'}
        
        current_profile_details['Recomendações'].append({
            'Rank': idx_rank + 1,
            'Título': recommended_film_row['title'],
            'Rating': recommended_film_row[NEW_SCORE_COLUMN],
            'Jaccard': f'{j_score:.4f}',
            'Features Válidas': features_do_filme
        })
    
    mean_jaccard_profile = np.mean(jaccard_scores_profile)
    current_profile_details['Jaccard Média do Perfil'] = mean_jaccard_profile
    detailed_results.append(current_profile_details)
    
    summary_table_data.append({
        'ID Perfil': idx_profile + 1,
        'Jaccard Média': f'{mean_jaccard_profile:.4f}',
        '% Coerência (Match)': f'{mean_jaccard_profile * 100:.2f}%'
    })

print("Avaliação de Coerência Concluída.")

mean_jaccard_coherence = np.mean(all_jaccard_scores)

print("\n" + "="*80)
print("I. RESULTADOS AGREGADOS DA COERÊNCIA DE FEATURES (MÉTRICA JACCARD)")
print("="*80)
print(f"Número de Perfis Testados: {n_profiles}")
print(f"Total de Recomendações Avaliadas: {len(all_jaccard_scores)}")
print("-" * 80)
print(f"Métrica de Coerência Jaccard Média: {mean_jaccard_coherence:.4f} (Quanto mais próximo de 1.0, melhor)")
print("="*80)

df_summary = pd.DataFrame(summary_table_data)

print("\n" + "="*80)
print("II. SUMÁRIO DA COERÊNCIA POR PERFIL (TOP 10)")
print("="*80)
print(df_summary.to_string(index=False))
print("="*80)