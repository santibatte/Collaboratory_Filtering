# Collaborative_Filtering
## Aprendizaje Maquina
### Proyecto Final

Santiago\
Patricia\
Jose Reyes 142207\
Yedam Fortiz 119523

Se implementará el algoritmo de filtrado colaborativo utilizando la metodología vista en clase:
* Utilizaremos el método de minimización alternada donde se optimiza primero una matriz respecto a otra que se definió aleatoriamente, para luego minimzar la segunda dada la primera
* Obtendremos k optima para el modelo considerando la métrica de desempeño de NDCG (https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)

Se utilizarán los archivos con el conjunto pequeño de calificaciones y películas ubicado en la siguiente liga https://www.kaggle.com/rounakbanik/the-movies-dataset:
*	links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset
*	ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies
Utilizaremos datos relevantes contenidos en los archivos:
*	movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies
*	keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object
*	credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object

#### Instrucciones de Ejecución:

El algoritmo de filtrado colaborativo se encuentra programando en un Jupyter Notebook Entrega_ProyectoFinal.ipynb. 
Para ejecutarlo, es necesario ejecutar todo el Notebook de forma secuencial. Se ha programado para calcular una matríz de recomendaciones 
para 100 usuarios distintos. El algoritmo puede tardar unos minutos en ejecutar. 

La última celda, contiene una función llamada Recomendar(). Esta función considera una matríz de recomendación para los 100 usuarios y las 9,000 películas de la base de datos.
Al ejecutar esta función (sin argumentos), se desplegará un cuadro de texto, en el cual debe introducirse el número de usuario (número entero del 1 al 100). Al dar enter, 
se desplegarán las 5 películas recomendads, con su título, fecha de lanzamiento y una breve sinópsis en inglés. Para hacer una nueva consulta, debe ejecutarse la función nuevamente.
