# Clasificador-Tipos-Pokemon

- Pokedex_Cleaned.csv : Archivo que contiene el nombre y tipo de cada pokémon a usar
- fetchPokemonDataset.py : Archivo que contiene un programa en python que guarda localmente el dataset que utilizaremos para la clasificación, fuente de los datos en [Créditos](#Créditos)
- fetchPokemonImages.py : Archivo que contiene un programa en python que guarda localmente las imagenes del dataset que utilizaremos para la clasificación, fuente de los datos en [Créditos](#Créditos)

## Métodos de clasificación

### Old-School Colors + SIFT + Bag of Words

- Clasificador_Old-School.py: Archivo que contiene diferentes modelos usados para la clasificación de los pokemon haciendo uso de métodos old-school como son la obtención de colores con KMeans, la obtención de características de las imagenes con SIFT y con este conjunto formamos nuestro Bag of Words listo para clasificar
- colores.csv: Archivo que contiene el dataset de Pokedex_Cleaned.csv pero balanceado y contando con los colores RGB que KMeans encuentra por cada imagen
- bow_dictionary.pkl: Archivo que contiene el diccionario que se utiliza para nuestra implementación del bag of words

### CNN

- Clasificador CNN.py : Archivo que contiene el codigo realizado para el clasificador usando un CNN
- cnn.pt : Archivo que contiene el modelo de CNN ya entrenado, con Train Accuracy = 87% Test Accuracy = 87% entrenado con 20 epocas y learning rate = 0.1

## Autoría

- Javier Alejandro Camacho Machaca | 1566088
- Abel Blanco Prous | 1606521
- Iván Peñarando Martínez | 1599156

# Créditos
[Fuente del dataset/imágenes <-> pokemondb.net](https://pokemondb.net/)