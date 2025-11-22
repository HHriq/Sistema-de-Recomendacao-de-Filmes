# 🎬 Sistema de Recomendação de Filmes - Projeto A3

**Instituição:** Unifacs - Universidade Salvador  
**Curso:** Inteligência Artificial  
**Professor:** Adailton de Jesus Cerqueira Junior  
**Semestre:** 2025.2  

<br>

## 📋 Descrição do Projeto

Este projeto consiste na implementação de um **Sistema de Recomendação de Filmes** baseado na abordagem de **Filtragem Baseada em Conteúdo (Content-Based Filtering)**.

A aplicação utiliza processamento de dados vetoriais para identificar padrões de similaridade entre o catálogo de filmes e as preferências especificadas pelo usuário. O sistema transforma atributos categóricos em vetores numéricos e calcula a proximidade entre eles em um espaço multidimensional.

<br>

## ⚙️ Arquitetura e Metodologia

O pipeline de processamento da recomendação foi estruturado em três etapas principais:

### 1. Pré-processamento e Vetorização (One-Hot Encoding)

- As variáveis categóricas (**Gênero, Diretor, Atores**) foram tratadas usando **One-Hot Encoding**, por ser mais adequado para cálculos de distância.
- O **Label Encoder** foi evitado, pois cria uma hierarquia ordinal artificial entre categorias.
- O One-Hot produz um vetor binário independente para cada categoria, garantindo que a similaridade reflita apenas presença/ausência da característica.
- O dataset foi dividido em **80% treino** e **20% teste**.

<br>

### 2. Ponderação de Atributos (Weighted Features)

O algoritmo aplica pesos diferentes conforme a relevância dos atributos:

| Nível de Relevância | Peso | Atributos |
|--------------------|-------|-----------|
| **Alta** | 3.0 – 2.5 | Ator Principal, Diretor |
| **Média** | 2.0 – 1.8 | Gênero, Temática |
| **Contexto** | 1.5 – 0.8 | Década, Produtora, País |

<br>

### 3. Pesos Dinâmicos (Dynamic Weight Adjustment)

A função **`ajustar_pesos_por_input`** adapta o cálculo dos pesos:

- Se o usuário escolher **"Outros"**, o peso daquela feature é zerado.
- Isso elimina a influência daquela dimensão no cálculo da similaridade.
- Garante que apenas preferências realmente informadas impactam o resultado.

<br>

### 4. Cálculo da Similaridade

A recomendação final utiliza a **Similaridade de Cosseno (cosine_similarity)**, que mede o ângulo entre:

- vetor do usuário (preferências)
- vetor de cada filme no dataset

Quanto mais próximo de 1, mais similar o filme.

<br>

## 🧪 Validação e Métricas de Desempenho

A avaliação, no script **avaliacaoModelo.py**, utiliza:

### 1. Coerência de Jaccard (Logical Coherence)

- Mede a interseção entre features solicitadas (input) e presentes nos filmes recomendados.
- **Resultado:** ~**25.13%**
- Perfis comuns atingiram **>40%** devido à maior densidade de vetores similares.

### 2. Curva ROC e AUC

- Avalia a capacidade do sistema de priorizar filmes relevantes (Rating ≥ 7.0).
- **AUC obtido:** **0.5854**
- Interpretação: ~59% de chance de ranquear um item relevante acima de um irrelevante.

<br>

## 👥 Equipe de Desenvolvimento

- Glenda Souza Fernandes dos Santos  
- Paulo Henrique Pereira Araujo Piedade  
- João Luccas Lordelo Marques  
- Marcus Vinicius Lameu Lima  
- Isaac Oliveira Dias  

<br>

## 📂 Estrutura do Repositório 

```
/
├── README.md
├── codigo_fonte/
│   ├── app.py
│   ├── treinamento_modelo_filmes.py
│   ├── logica_recomendacao.py
│   ├── avaliacaoModelo.py
│   ├── dataset_tratado - Filmes.csv
│   ├── estilos.css
│   └── requirements.txt
└── poster/
    └── sistema_de_recomendacao_de_filmes_banner.pdf
```

<br>

## 🚀 Guia de Instalação e Execução

### 1. Instalar Dependências

```bash
git clone https://github.com/HHriq/Sistema-de-Recomendacao-de-Filmes.git
cd Sistema-de-Recomendacao-de-Filmes
pip install -r codigo_fonte/requirements.txt
```

<br>

### 2. Treinar o Modelo (Obrigatório)

```bash
cd codigo_fonte
python treinamento_modelo_filmes.py
```

##### Após finalizar, você verá:

```
✅ Modelo salvo com sucesso!
```

<br>

### 3. Executar a Aplicação

```bash
streamlit run app.py
```

<br>

### 4. Executar Relatório de Métricas (Validação do Modelo - Opcional)


Este script executa uma bateria de testes automatizados no terminal, sem interface gráfica. Ele submete **perfis sintéticos** (usuários simulados com gostos específicos) ao modelo para validar se as recomendações seguem a lógica esperada.


**O que será exibido:**

Ao executar o comando, o terminal gerará um relatório estatístico contendo:

1.  **Coerência de Jaccard:** A porcentagem média de características (ex: Ator, Gênero) que o modelo acertou nas recomendações.

2.  **Tabela de Performance:** O desempenho individual de cada perfil de teste.

3.  **Métricas Globais:** O resumo da assertividade do sistema.

**Execute no terminal:**

```bash
python avaliacaoModelo.py
```

<br>

## © 2025 - Inteligência Artificial
