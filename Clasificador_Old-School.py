# Dataset
from sklearn import datasets
# Data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.cluster import KMeans
import pickle
from scipy.spatial.distance import cdist
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

rutaProyecto = os.path.abspath('')  # Carpeta general donde se encuentra el proyecto
rutaImgs = rutaProyecto + "\images" # Carpeta que contiene las imagenes de pokémon a aumentar (data augmentation) y usar

# Carga del dataset
missing_values = ["n/a", "na", "--","-"]

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', na_values = missing_values)
    return dataset

DataFrame = load_dataset(rutaProyecto + '/Pokedex_Cleaned.csv')

# Creación de la variable DataFrame_final donde se filtran 7 tipos 
# ('Grass', 'Fire', 'Water', 'Normal', 'Bug', 'Psychic', 'Others')
# Con el color principal de cada pokemon
if os.path.exists(rutaProyecto + '/colores.csv'): # Importa DataFrame_final con el filtrado aplicado y los colores. Columnas: ['name', 'type_1', 'img_name', 'blue', 'green', 'red']
  DataFrame_final = load_dataset(rutaProyecto + '/colores.csv')
  DataFrame_final = DataFrame_final.drop('Unnamed: 0', axis=1)
else:
# Filtramos el dataset para usar los tipos más frecuentes
    types = DataFrame['type_1'].drop_duplicates()
    df_names = DataFrame['name']
    num = []
    for i in types:
        num_pkm = df_names[DataFrame['type_1'].str.contains(i)].values.size
        num.append([i, num_pkm])

    num_filter = []
    num_filter_erase = []
    min_num = 75

    for i in num:
        if i[1]>=min_num:
            num_filter.append(i)
        else:
            num_filter_erase.append(i)

    print("Tipos: ", num_filter)
    print("Tipos filrados: ", num_filter_erase)

    total_num_filter_erase = 0

    for i in num_filter_erase:
        total_num_filter_erase += i[1]

    print("Número de pokémon en Others: ", total_num_filter_erase)

    DataFrame_final = DataFrame
    DataFrame_filter = DataFrame

    for i in num_filter_erase:
        DataFrame_final = DataFrame_final[DataFrame['type_1'].str.contains(i[0])==0]    

    for i in num_filter:
        DataFrame_filter = DataFrame_filter[DataFrame['type_1'].str.contains(i[0])==0]

    DataFrame_filter['type_1'] = 'Others'


    # Eliminamos las filas del dataset sin carpeta dentro de images (Es decir, que no tienen imágenes)
    # El dataframe filtrado queda en la variable DataFrame_final
    def noDisponibles(nom, ruta, noDisp):
        rutaPkmn = ruta + '/' + nom
        if not os.path.exists(rutaPkmn):
            noDisp.append(nom)

    noDisp = []
    for poke in DataFrame_final['name']:
        noDisponibles(poke, rutaImgs, noDisp)

    filasEliminar = DataFrame_final['name'].apply(lambda x: x not in noDisp)
    DataFrame_final = DataFrame_final.loc[filasEliminar]

    noDisp = []
    for poke in DataFrame_filter['name']:
        noDisponibles(poke, rutaImgs, noDisp)

    filasEliminar = DataFrame_filter['name'].apply(lambda x: x not in noDisp)
    DataFrame_filter = DataFrame_filter.loc[filasEliminar]

    DataFrame_final = pd.concat([DataFrame_final, DataFrame_filter])
    DataFrame_final = DataFrame_final.reset_index(drop=True)

    df_orig = DataFrame_final

    for poke_row in df_orig.iterrows():
        if (len(DataFrame_final[DataFrame_final['name'] == poke_row[1]['name']]) > 1):
            print(f"Dropping {poke_row[1]['name']}")
            DataFrame_final = DataFrame_final[DataFrame_final['name'] != poke_row[1]['name']]

    DataFrame_final['img_name'] = "0.jpg"


    # Balanceamos el dataframe asegurandonos que cada clase tenga el mismo número
    # Deja el resultado en la variable DataFrame_final
    # Creamos nuevas filas utilizando las imagenes extra del pokemon,
    # guardando el nombre de la imagen (Nos servira más adelante para extraer el color)
    original_dataset = DataFrame_final

    max_rows_per_type = 185 # Fija el número por clase
    max_size = max_rows_per_type * DataFrame_final.type_1.unique().shape[0]

    while(DataFrame_final.shape[0] != max_size):
        last_iteration_size = DataFrame_final.shape[0]

        DataFrame_final_orig = DataFrame_final

    for poke_row in DataFrame_final_orig.iterrows():
        current_type = poke_row[1]['type_1']
        current_poke = poke_row[1]['name']

        if (max_rows_per_type == len(DataFrame_final[DataFrame_final['type_1'] == current_type])):
            continue
        
        img_pokes = len(DataFrame_final[DataFrame_final['name'] == poke_row[1]['name']])
        max_imgs = len(os.listdir(rutaImgs + f'/{current_poke}'))
        
        if (max_imgs <= img_pokes):
            continue
    
        new_row = poke_row[1]
        new_row['img_name'] = f"{img_pokes}.jpg"

        DataFrame_final = pd.concat([DataFrame_final, pd.DataFrame(new_row).transpose()])

        if (DataFrame_final.shape[0] == last_iteration_size):
            print("Not enough images, change max_rows_per_type to correctly balance dataset")
            print("Replacing dataset with original one before balancing, unbalanced one saved at unbalanced_dataset")

            unbalanced_dataset = DataFrame_final
            DataFrame_final = original_dataset

            break
    else:
        print("Dataset balanced!")


    # Por cada pokémon del dataframe, obtenemos sus imagen y extraemos su color principal
    bgr_colors = [] # Array con los colores principales de las imagenes

    for pokemon_row in tqdm(DataFrame_final.iterrows()):
        pokemon_row = pokemon_row[1].transpose()
        pokemon_act = pokemon_row['name']
        img_act = pokemon_row['img_name']
        imagen = cv2.imread(rutaImgs + f'/{pokemon_act}/{img_act}')

        # aplicamos kmeans con criteria y numero de clusters (K)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 3
        _, label, center = cv2.kmeans(np.float32(imagen).reshape(-1, 3), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convertimos a uint8 de nuevo y hacemos reshape para tener la imagen original con los k colores definidos
        res = np.uint8(center)[label.flatten()]
        res2 = res.reshape(imagen.shape)

        colors, count = np.unique(res2.reshape(-1, res2.shape[-1]), axis=0, return_counts=True)
        bgr_colors.append(colors[1])


    # Añadimos los colores en sus respectivas filas del dataframe
    bgr_colors = np.array(bgr_colors)

    DataFrame_final['blue'] = bgr_colors[:,0]
    DataFrame_final['green'] = bgr_colors[:,1]
    DataFrame_final['red'] = bgr_colors[:,2]


# Extraemos las características de las imagenes con SIFT

# Funcion para rellenar con 0 las filas de SIFT para tener la misma longitud
def ampliarSIFT(fila, max_length):
    if len(fila) < max_length:
        return np.pad(fila, (0, max_length - len(fila)), mode='constant')
    else:
        return fila

# Funciones para guardar SIFT en el DataFrame
# Construye un Bag of Words 
def kmean_bow(all_descriptors, num_cluster):
    bow_dict = []

    kmeans = KMeans(n_clusters = num_cluster)
    kmeans.fit(all_descriptors)

    bow_dict = kmeans.cluster_centers_

    if not os.path.isfile('/content/drive/MyDrive/proyecto visión por computador/Gen 1 to 9 images/bow_dictionary.pkl'):
        new_file = open('/content/drive/MyDrive/proyecto visión por computador/Gen 1 to 9 images/bow_dictionary.pkl', 'wb')
        pickle.dump(bow_dict, new_file)
        new_file.close()

    return bow_dict

# Generar las características BoW de las imagenes
def create_feature_bow(image_descriptors, BoW, num_cluster):

    X_features = []

    for i in tqdm(range(len(image_descriptors))):
        features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)

            argmin = np.argmin(distance, axis = 1)

            for j in argmin:
                features[j] += 1
        X_features.append(features)

    return X_features

# Extrae descriptores de cada imagen
sift_descriptors = []

for pokemon_row in tqdm(DataFrame_final.iterrows()):
  pokemon_row = pokemon_row[1]
  pokemon_act = pokemon_row['name']
  img_act = pokemon_row['img_name']

  imagen = cv2.imread(rutaImgs + f'/{pokemon_act}/{img_act}')

  sift = cv2.xfeatures2d.SIFT_create()
  kp1, des = sift.detectAndCompute(imagen, None)

  sift_descriptors.append(des)

# Los junta todos en un array
all_descriptors = []
for descriptor in tqdm(sift_descriptors):
    if descriptor is not None:
        for des in descriptor:
            all_descriptors.append(des)

# Crea el diccionario BoW (Bag of Words) que se utiliza como una representación compacta de las características visuales de las imágenes
num_cluster = 100
# Esto carga la variable BoW para no tener que realizar el proceso cada vez, ya que tarda bastante (BoW hecha con 100 clusters)
if not os.path.exists(rutaProyecto + '/bow_dictionary.pkl'):
    BoW = kmean_bow(all_descriptors, num_cluster)
else:
    BoW = pickle.load(open(rutaProyecto + '/bow_dictionary.pkl', 'rb'))

# Se utiliza para generar las características BoW (Bag of Words) de las imágenes en función de los descriptores de imagen y el diccionario BoW.
X_features = create_feature_bow(sift_descriptors, BoW, num_cluster)

# Creamos X e y
X = DataFrame_final[['red', 'green', 'blue']] 
siftFeatures = pd.DataFrame({'sift': X_features})
siftFEAT = siftFeatures['sift'].apply(ampliarSIFT, max_length=siftFeatures['sift'].apply(len).max())
expanded_sift = pd.DataFrame(siftFEAT.tolist())
X = pd.concat([X[['red', 'green', 'blue']], expanded_sift], axis=1)
X.columns = X.columns.astype(str)
y = DataFrame_final[['type_1']]

# Dejamos las particiones hechas para los clasificadores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Pruebas realizadas con RandomForest y KNN

# Prueba con RandomForest
param_grid = { 
    'n_estimators': [200, 500],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rfc=RandomForestClassifier(random_state=42)

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, verbose = 3)
CV_rfc.fit(X_train, y_train.values.ravel())

# Calcular la presición
accuracy = CV_rfc.score(X_test, y_test)

# print best parameter after tuning
print(CV_rfc.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(CV_rfc.best_estimator_)

# print best accuracy after hyper-parameter tuning
print(f"Best accuracy was: {CV_rfc.best_score_}")

###########################################################

# Prueba con KNN
param_grid = dict(n_neighbors=list(range(1, 100)))

knn = KNeighborsClassifier()

# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=5, verbose=3)
  
# fitting the model for grid search
grid_search=grid.fit(X_train, y_train.values.ravel())

# Calcular la presición
accuracy = grid_search.score(X_test, y_test)

# print best parameter after tuning
print(grid_search.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid_search.best_estimator_)

# print best accuracy after hyper-parameter tuning
print(f"Best accuracy was: {grid_search.best_score_}")

###########################################################

# Pruebas con SVM (mejores resultados)
# defining parameter range
param_grid = {'C': loguniform(1e0, 1e3), 
              'gamma': loguniform(1e-4, 1e1),
              'kernel': ['rbf']} 

# Entrenar el modelo SVM
clf = RandomizedSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1, cv = 5, n_iter = 300)

clf.fit(X_train, y_train.values.ravel())

# Calcular la presición
accuracy = clf.score(X_test, y_test)

# print best parameter after tuning
print(clf.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(clf.best_estimator_)

# print best accuracy after hyper-parameter tuning
print(f"Best accuracy was: {clf.best_score_}")

###########################################################

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['sigmoid','rbf']} 

# Entrenar el modelo SVM
clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, n_jobs=-1)

clf.fit(X_train, y_train.values.ravel())

# Calcular la presición
accuracy = clf.score(X_test, y_test)

# print best parameter after tuning
print(clf.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(clf.best_estimator_)

# print best accuracy after hyper-parameter tuning
print(f"Best accuracy was: {clf.best_score_}")
