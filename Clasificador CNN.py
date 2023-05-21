# Imports varios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import time
import random
from PIL import Image

# Imports para el CNN
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
import torch.nn as nn
from torch import flatten
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms


# Rutas usadas frecuentemente
rutaProyecto = os.path.abspath('')  # Carpeta general donde se encuentra el proyecto
rutaImgs = rutaProyecto + "\images" # Carpeta que contiene las imagenes de pokémon a aumentar (data augmentation) y usar


# Carga del dataset
missing_values = ["n/a", "na", "--","-"]

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', na_values = missing_values)
    return dataset

DataFrame = load_dataset(rutaProyecto + '/Pokedex_Cleaned.csv')


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


############################################################################
# DATA AUGMENTATION                                                        #
# Codigo usado para augmentar el número de imagenes y igualar las 7 clases #
############################################################################

maxValor = 3000 # Número de imágenes que tendra cada clase

# Calculo de variables necesarias para aumentar los datos y declaración de funciones

diccValors = {'Grass': 0, 'Fire': 0, 'Water': 0, 'Bug': 0, 'Normal': 0, 'Psychic': 0, 'Others': 0} # Diccionario que contendrá el número de imagenes actuales de cada clase (usado para el aumento)
nomsAugmentar = {} # Nombre de todas las carpetas con su tipo (usado para el aumento)
df = pd.DataFrame(columns=["type_1", "img"])
ruta = rutaImgs
progress_bar = tqdm(total=DataFrame_final.shape[0], desc="Calculo de nomsAugmentar y diccValors")
for _, poke in DataFrame_final.iterrows():
  nombrePkmn = poke['name']
  tipoPkmn = poke['type_1']
  rutaPkmn = ruta + '/' + nombrePkmn
  if os.path.exists(rutaPkmn):
    arxius = os.listdir(rutaPkmn)
    nomsAugmentar[nombrePkmn] = tipoPkmn
    diccValors[tipoPkmn] += len(arxius) # Guarda el número de pokémon de cada tipo que hay actualmente
  progress_bar.update(1)
progress_bar.close()

# Función usada para guardar una imagen sin sobreescribir otra
def escribirImgSinRepetir(img, nombre, ruta):
    i = 0
    filename = nombre + ".jpg"
    while os.path.isfile(os.path.join(ruta, filename)):
        # Si el archivo ya existe, agregar un sufijo al nombre
        i += 1
        filename = "{}_{}.jpg".format(nombre, i)

    img.save(os.path.join(ruta, filename))

# Función que genera y guarda una imagen aleatoria, recortando una imagen dada y aplicando una simetria vertical/horizontal aleatoriamente
def dataAugm(image, nombre, ruta):
  width, height = image.size

  crop_width = width - 70
  crop_height = height - 70

  x = random.randint(0, width - crop_width)
  y = random.randint(0, height - crop_height)
  cropped_image = image.crop((x, y, x + crop_width, y + crop_height))
  random_number = random.randrange(3)

  if random_number == 0:
    flipped_image = cropped_image.transpose(Image.FLIP_TOP_BOTTOM)
  else:
    if random_number == 1:
      flipped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
      flipped_image = cropped_image

  escribirImgSinRepetir(flipped_image, nombre, ruta)

# Función que controla el bucle de data augmentation
# Comprueba que todas las clases hayan llegado al valor máximo
def comprovar(maxValor, diccValors):
 numBe = 0
 for clave, valor in diccValors.items():
   if valor >= maxValor:
     numBe += 1

 if numBe == 7:
   return True
 else:
   return False

# Bucle de aumento de imagenes
arxius = os.listdir(rutaImgs) # Lista de nombres de las carpetas a aumentar
final = False # Variable que controla el bucle, según la función "comprovar"
while(final == False):
  for carpeta in arxius:
    nombre = carpeta
    if nombre in nomsAugmentar: 
      tipoPkmn = nomsAugmentar[nombre]
      if diccValors[tipoPkmn] < maxValor: # Si algún tipo llega al tope, no aumentamos más
        rutaCarpeta = rutaImgs + '/' + carpeta
        if os.path.exists(rutaCarpeta):
          image = Image.open(rutaCarpeta + '/0.jpg') # Aumentamos en base a la primera imagen solo ya que hay pokemon que solo tienen 1
          dataAugm(image, nombre, rutaCarpeta) # Aumentamos la imagen y la guardamos
          diccValors[tipoPkmn] += 1
          if (diccValors[tipoPkmn] == maxValor):
            print("Acabado aumento de tipo: ", tipoPkmn)
    final = comprovar(maxValor, diccValors)
    if final == True:
      break



# Al tener las imagenes aumentadas, las cargamos dentro de un dataset final
if os.path.exists(rutaProyecto + '/dataframeImgs.pkl'): # Usado para cargar el dataframe directamente con las imagenes
  with open(rutaProyecto + '/dataframeImgs.pkl', 'rb') as file:
    df = pickle.load(file)
else:
    df = pd.DataFrame(columns=["type_1", "img"]) # Creamos dataframe que contendra la imagen y el tipo del pokémon
    progress_bar = tqdm(total=DataFrame_final.shape[0], desc="Cargando dataframe con imagenes")

    for _, poke in DataFrame_final.iterrows():
        nombrePkmn = poke['name']
        tipoPkmn = poke['type_1']
        rutaPkmn = rutaImgs + '/' + nombrePkmn
        if os.path.exists(rutaPkmn):
            arxius = os.listdir(rutaPkmn)
            for arxiu in arxius:
                rutaImg = rutaPkmn + '/' + arxiu
                with Image.open(rutaImg) as img:
                    novaFila = {"type_1": tipoPkmn, "img": img.copy()}
                    df = pd.concat([df, pd.DataFrame([novaFila])], ignore_index=True)
        progress_bar.update(1)
    progress_bar.close()


############
#   CNN    #
############

# Definición del CNN usado para la clasificación de tipos a partir de imagenes
class CNN(nn.Module):
    def __init__(self, num_classes): # En num clases se incluye el numero de clases a clasificar, en nuestro caso 7
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=4, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv_layer4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv_layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(256, num_classes)
      
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)

        out = self.conv_layer3(out)
        out = self.relu3(out)

        out = self.conv_layer4(out)
        out = self.relu4(out)

        out = self.conv_layer5(out)
        out = self.relu5(out)
        out = self.max_pool5(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.dropout6(out)
        out = self.fc6(out)
        out = self.relu6(out)

        out = self.dropout7(out)
        out = self.fc7(out)
        out = self.relu7(out)

        out = self.fc8(out)
        return out

# PREPARACIÓN DE LOS DATOS PARA INTRODUCIRLOS AL MODELO

# Habilitamos la posibilidad de usar cuda si esta disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creamos una equivalencia para los tipos para introducirlos en el modelo (De str a int)
dicc_tipos = {'Grass': 0, 'Fire': 1, 'Water': 2, 'Bug': 3, 'Normal': 4, 'Psychic' : 5, 'Others' : 6}
df['type_1'] = df['type_1'].map(dicc_tipos)

# Definimos las transformaciones de las imagenes (Reducimos el tamaño de las imagenes para que vaya más rápido y las convertimos a tensor)
img_size = 32 
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

print("Preprocesamiento de imagenes")

# Creamos un train y test set, para entrenar el modelo y testearlo
from sklearn.model_selection import train_test_split
X = df['img']
y = df['type_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

input_trains = [] # Donde se guardan los tensors de las imagenes de train procesadas

progress_bar = tqdm(total=train.shape[0], desc="Preprocesamiento de imagenes de train")
for _, fila in train.iterrows():
  img = fila['img']
  tipo = fila['type_1']
  if img.mode != 'RGB':
    img = img.convert("RGB")
  tensor = preprocess(img)
  input_trains.append([tensor, tipo])
  progress_bar.update(1)
progress_bar.close()

# Calculamos mean y standard deviation para poder normalizar las imagenes procesadas
np.random.seed(0)
idx = np.random.randint(0, len(input_trains), 512)
tensors = torch.concat([input_trains[i][0] for i in idx], axis=1)
tensors = tensors.swapaxes(0, 1).reshape(3, -1).T
mean = torch.mean(tensors, axis=0)
std = torch.std(tensors, axis=0)
del tensors

# Normalizamos las imagenes procesadas de train
preprocess = transforms.Compose([
    transforms.Normalize(mean=mean, std=std)
])

for i in tqdm(range(len(input_trains))):
  input_tensor = preprocess(input_trains[i][0])
  input_trains[i][0] = input_tensor

# Definimos nuevamente las operaciones de preproceso incluyendo la normalización y preprocesamos las imagenes de test
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

input_tests = []

progress_bar = tqdm(total=test.shape[0], desc="Preprocesamiento de imagenes de test")
for _, fila in test.iterrows():
  img = fila['img']
  tipo = fila['type_1']
  if img.mode != 'RGB':
    img = img.convert("RGB")
  tensor = preprocess(img)
  input_tests.append([tensor, tipo])
  progress_bar.update(1)
progress_bar.close()

# Definimos el tamaño del lote en el que se cargaran las imagenes
batch_size = 64
# Ponemos todo en DataLoaders
dLoader_train = torch.utils.data.DataLoader(
    input_trains, batch_size=batch_size, shuffle=True
)

dLoader_test = torch.utils.data.DataLoader(
    input_tests, batch_size=batch_size, shuffle=True
)

print("Resultados CNN: ")

# Si no hay modelo guardado, volvemos a entrenar el modelo
if not os.path.exists(rutaProyecto + '/cnn.pt'): 
    # Indicamos número de clases y enviamos el model a device
    num_classes = 7
    model = CNN(num_classes).to(device)
    # Funcion loss
    loss_func = nn.CrossEntropyLoss()
    # Learning rate 
    lr = 0.1  # 0.01
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    # Empezamos el entrenamiento del modelo
    num_epochs = 20 # Definimos el número de epocas
    for epoch in range(num_epochs):
        st = time.time()
        model.train()
    # Cargamos los datos por lotes (De tamaño definido en batch_size)
        for i, (images, labels) in enumerate(dLoader_train):  
            # Movemos tensores a device
            images = images.to(device)
            labels = labels.to(device)
            
            # forward propagation
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
            # backward propagation and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            all_val_loss = []
            for images, labels in dLoader_test:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total += labels.size(0)
                # Calculamos prediciones
                predicted = torch.argmax(outputs, dim=1)
                # Calculamos valores reales
                correct += (predicted == labels).sum().item()
                # Calculamos loss
                all_val_loss.append(loss_func(outputs, labels).item())
            # Calculamos val-loss
            mean_val_loss = sum(all_val_loss) / len(all_val_loss)
            # Calculamos val-accuracy
            mean_val_acc = 100 * (correct / total)
        en = time.time()
        print(
            'Epoch [{}/{}], Loss: {:.4f}, Val-loss: {:.4f}, Val-acc: {:.1f}%'.format(
                epoch+1, num_epochs, loss.item(), mean_val_loss, mean_val_acc
            )
        )
        print("Epoch time: ", en-st)
else:
   # Cargamos y evaluamos la accuracy de train del modelo
   if device.type == "cuda":
    model = torch.load('cnn.pt')
   else:
    model = torch.load('cnn.pt', map_location=torch.device('cpu'))

   model.eval() 

   totalTrain = 0
   correctosTrain = 0

   for images, labels in dLoader_train:
        images = images.to(device)
        labels = labels.to(device)

        # Calculo de las predicciones del modelo
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        totalTrain += labels.size(0)
        correctosTrain += (predicted == labels).sum().item()

        # Calcula la accuracy de train
        accuracyTrain = 100 * correctosTrain / totalTrain

   print('Train Accuracy: {:.2f}%'.format(accuracyTrain))
   

# Calculo de accuracy de test
input_tensors = []

for image in test['img']:
    tensor = preprocess(image)
    input_tensors.append(tensor.to(device))

# Calculo de las predicciones del modelo
input_tensors = torch.stack(input_tensors)
outputs = model(input_tensors)
predicted = torch.argmax(outputs, dim=1)

predicciones = predicted.detach().cpu().numpy()
real = test['type_1']

accuracyTest = (np.sum(predicciones == real) / real.shape)[0]
print('Test Accuracy {:.2f}%'.format(accuracyTest * 100))