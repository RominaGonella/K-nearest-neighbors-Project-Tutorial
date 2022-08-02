# K-nearest neighbors Project Tutorial

# librerías
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# cargo datos del repositorio
movies = pd.read_csv('data/raw/tmdb_5000_movies.csv')
credits = pd.read_csv('data/raw/tmdb_5000_credits.csv')

# se unen ambos datasets
movies = movies.merge(credits, on = 'title')

# sólo variables de interés
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# se eliminan 3 filas con missing values
movies.dropna(inplace = True)

# función para traer el contenido de name de variable "genre"
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# se aplica función a "genres" y a "keywords"
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# función para traer los 3 primeros elementos de name en una columna
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

# aplico función a "cast"
movies['cast'] = movies['cast'].apply(convert3)

# función para traer el nombre del director
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# se aplica función
movies['crew'] = movies['crew'].apply(fetch_director)

# se convierte columna overview a una lista
movies['overview'] = movies['overview'].apply(lambda x : x.split())

# función para quitar espacios entre palabras
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# se aplica función a varias columnas
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# se crea nueva variable con información de 5 previas
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

# se crea nuevo dataset con 3 variables
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

# se crea vectorizador para un máximo de 5000 palabras
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# se calcula similaridad entre películas
cosine_similarity(vectors).shape
similarity = cosine_similarity(vectors)

# función que identifica las 5 películas más similares a movie (sin contarla a ella misma)
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
