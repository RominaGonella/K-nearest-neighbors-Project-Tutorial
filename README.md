# Resumen del proceso

1. El objetivo de la tarea consiste en construir un sistema de recomendación de películas sencillo utilizando KNN.
2. Se utilizan dos datasets: uno de películas y otro de créditos de dichas películas, ambos provienen de  TMDB Movie Database API y contienen aproximadamente 5000 películas cada uno (2803). Contienen información sobre la película, como el título, género, palabras clave, director, etc. Para este trabajo se seleccionan solamente algunas de las variables.
3. Se realiza un procesamiento de los datos de interés, eliminando filas con missing values, extrayendo solamente la información deseada de los campos de tipo json, quitando espacios entre palabras, etc.
4. Se crea una variable que concatena las 5 variables a utilizar: overview, genres, keywords, cast y crew. Sobre esta variable se aplica la vectorización y luego se calcula el cosine_similarity, que mide la similaridad entre las distintas películas (de acuerdo a todas las variables en simultáneo).
5. Se crea función de recomendaciones utilizando el resultado del cosine similarity, para cada input (película) devuelve las 5 películas más similares a ella.
6. Se prueba el sistema con algunas películas elegidas al azar, y para la famosa película Titanic.