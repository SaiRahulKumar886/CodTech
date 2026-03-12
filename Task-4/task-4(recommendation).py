import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import gc
from IPython.display import Markdown
import pickle
from scipy.sparse import csr_matrix
np.random.seed(3213)
ratings = pd.read_csv('ratings.csv', engine="pyarrow")
ratings.head()
movies = pd.read_csv("movies.csv", engine="pyarrow")
movies.head()
ratings.dtypes
ratings['userId'] = ratings['userId'].astype("int32")
ratings['movieId'] = ratings['movieId'].astype("int32")
ratings['rating'] = ratings['rating'].astype("float32")

# Essa coluna sem significância relevante também pode comprometer computacionalmente nossas análises por isso iremos removê-la
ratings.drop('timestamp', axis=1, inplace=True)
print(ratings.dtypes)
# Filtrar usuários e filmes com pelo menos n avaliação
gc.collect()
df_before = len(ratings)
user_counts = ratings['userId'].value_counts()
movie_counts = ratings['movieId'].value_counts()
filtered_ratings = ratings[
    ratings['userId'].isin(user_counts[user_counts >= 150].index) &
    ratings['movieId'].isin(movie_counts[movie_counts >= 250].index)
]
df_after = len(filtered_ratings)
print("Tamanho do Dataset antes da filtragem: {:d}. Tamanho após a filtragem: {:d}. Redução de {:d} registros".
     format(df_before, df_after, (df_before - df_after)))
filtered_ratings.head(5)
gc.collect()
user_movie_matrix = filtered_ratings.pivot_table(index='userId',
                                                 aggfunc='mean', columns='movieId', values='rating').fillna(0)
gc.collect()
sparse_matrix = csr_matrix(user_movie_matrix)
gc.collect()
sparse_similarity = cosine_similarity(sparse_matrix, dense_output=False)
# Remove a similaridade entre um usuário e ele mesmo
sparse_similarity.setdiag(0)
sparse_similarity.eliminate_zeros()
rows, cols, sims = [], [], []

for i in range(sparse_similarity.shape[0]):
    gc.collect()
    row = sparse_similarity.getrow(i)
    if row.nnz == 0:
        continue
    top_indices = row.data.argsort()[::-1][:3]  # top-3 maiores
    top_cols = row.indices[top_indices]
    top_sims = row.data[top_indices]

    rows.extend([i]*len(top_cols))
    cols.extend(top_cols)
    sims.extend(top_sims)

df_top3 = pd.DataFrame({
    "userId": rows,
    "similarUserId": cols,
    "similarity": sims
})
try:
    del user_movie_matrix, sparse_matrix
except:
    print("Objetos não encontrados no ambiente")
df_top3.loc[df_top3['userId'].isin([12, 13])]
gc.collect()
similar_users = df_top3['similarUserId'].unique()
similar_user_ratings = filtered_ratings[filtered_ratings['userId'].isin(similar_users)]
similar_user_ratings.sort_values(by=["userId", "rating"], ascending=[True, False]).head(3)
# Ordena por userId e rating decrescente
sorted_ratings = similar_user_ratings.sort_values(by=["userId", "rating"], ascending=[True, False])

# Agrupa por usuário e pega os 3 primeiros de cada grupo
top3_by_user = sorted_ratings.groupby("userId").head(3).reset_index(drop=True)

# Exibe o resultado
print(top3_by_user)
movies
# Filtra o registro dos usuários 12 e 13
filtered = top3_by_user[top3_by_user['userId'].isin([12,13])]

# Pega o Id Dos filmes recomendados
movie_ids = filtered['movieId'].unique()

# Recupera os títulos recomendados
recommended_titles = movies[movies["movieId"].isin(movie_ids)]

recommended_titles
top3_by_user.to_parquet("top3_by_user.parquet", compression="snappy")
df_top3.to_parquet("user_similarity_top3.parquet", compression="snappy")
def recommend_movies(user_id, n=5):
    similar_users = df_top3[df_top3['userId'] == user_id]['similarUserId']

    movies_from_similar = filtered_ratings[
        filtered_ratings['userId'].isin(similar_users)
    ]

    top_movies = movies_from_similar.sort_values(
        by='rating', ascending=False
    ).head(n)

    recommended = movies[movies['movieId'].isin(top_movies['movieId'])]

    return recommended[['movieId','title']]
recommend_movies(12)
print("Recommended Movies for User 12:")
print(recommend_movies(12))
plt.hist(filtered_ratings['rating'], bins=10)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()