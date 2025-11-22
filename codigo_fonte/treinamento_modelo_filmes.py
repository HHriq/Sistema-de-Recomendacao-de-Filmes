import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Função: Converte número de prêmios, Sim / Não.

def converter_premios(valor):
    valor = int(valor)
    return "Sim" if valor >= 1 else "Não"
    
# Função de Treinamento 

def treinar_modelo_recomendacao(
        dataset_path,
        output_model_path="./modelo_recomendacao_ContentBased.joblib"
    ):

    print("📂 Carregando dataset...")
    df = pd.read_csv(dataset_path)

    # 1. Renomear colunas para nomes padrão 
    
    col_mapping = {
        "Nome do Filme": "title",
        "Ano de Lancamento": "release_year",
        "Decada do Filme": "release_decade",
        "Duracao": "duration",
        "Lancamento": "release_type",
        "Classificao do Filme": "movie_rating",
        "Publico Alvo": "age_rating",
        "Diretor": "director",
        "Estrela": "main_actor",
        "Genero": "genre",
        "Tematica": "theme",
        "Pais de Origem": "country",
        "Produtora": "producer",
        "Idioma": "language",
        "Indicado a Premiações de Cinema": "premios",
        "Nota": "rating"
    }
    df.rename(columns=col_mapping, inplace=True)

    # 2. Seleção de Features Relevantes

    features = [
        "genre", "theme", "main_actor", "director", "producer",
        "country", "age_rating", "release_decade", "release_type", "premios"
    ]

    df = df.dropna(subset=["title"])

    # 3. Tratar valores ausentes

    fill_map = {
        "genre": "Outros",
        "theme": "Outros",
        "main_actor": "Outros",
        "director": "Outros",
        "producer": "Outros",
        "country": "Outros",
        "age_rating": "Outros",
        "release_decade": "Outros",
        "release_type": "Outros",
        "premios": 0
    }

    df.fillna(fill_map, inplace=True)
    df["premios"] = df["premios"].apply(converter_premios)

    
    # 4. Separar treino e teste

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 5. One-Hot Encoding das features categóricas

    categorical_cols = features # Todas as features são categóricas após o tratamento

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )

    print("🔧 Convertendo features (OneHotEncoder)...")
    X_train = encoder.fit_transform(train_df[features])
    X_test = encoder.transform(test_df[features])

    # 6. Criar matriz de similaridade usando Cosine Similarity

    print("📐 Calculando matriz de similaridade (train)...")
    similarity_matrix = cosine_similarity(X_train, X_train)

    print("📐 Calculando matriz de similaridade (test)...")
    similarity_matrix_test = cosine_similarity(X_test, X_train)

    # 7. Salvar modelo e dados importantes

    print("💾 Salvando modelo...")

    model_data = {
        "encoder": encoder,
        "similarity_matrix_train": similarity_matrix,
        "similarity_matrix_test": similarity_matrix_test,
        "train_data": train_df.reset_index(drop=True),
        "test_data": test_df.reset_index(drop=True),
        "features": features
    }

    joblib.dump(model_data, output_model_path)

    print("\n✅ Modelo salvo com sucesso!")
    print(f"📦 Caminho: {os.path.abspath(output_model_path)}")
    print(f"🎬 Filmes usados no treino: {len(train_df)}")
    print(f"🎬 Filmes usados no teste: {len(test_df)}")

    return output_model_path



# Execução direta

if __name__ == "__main__":
    modelo_path = treinar_modelo_recomendacao("./dataset_tratado - Filmes.csv")
    print("\nModelo salvo em:", modelo_path)