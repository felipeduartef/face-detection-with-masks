import cv2
import os
import numpy as np

# Ruta del directorio
dataPath = "././Dataset_Train"
dir_list = os.listdir(dataPath)    #Lista de directorios en la posición donde se encuentra
print("Lista archivos: ", dir_list) 

labels = []     # Etiquetas para cada imagen de aucerdo a la clasificación a la que pertenecen
facesData = []  # Arreglo donde se almacena la información de cada imagen
label = 0       #etiqueta de clasificación

#For para la lectura de cada imagen del dataset en los dos directorios
for name_dir in dir_list:
     dir_path = dataPath + "/" + name_dir   #Ruta de los dos directorios
     
     #Lectura de cada imagen presente en el directorio
     for file_name in os.listdir(dir_path):

          image_path = dir_path + "/" + file_name #Ruta de la imagen

          image = cv2.imread(image_path, 0)       #Lectura de la imagen

          facesData.append(image)        #Almacenamiento de la información en el arreglo facesData
          labels.append(label)           #Etiqueta para la imagen
     label += 1

# LBPH FaceRecognizer
face_mask = cv2.face.LBPHFaceRecognizer_create()
# Entrenamiento
print("Entrenando...")
face_mask.train(facesData, np.array(labels))
# Almacenar modelo
face_mask.write("train/face_mask_model.xml")
print("Modelo almacenado")