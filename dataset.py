from itertools import count
import cv2
from matplotlib import pyplot as pt
import imutils
import mediapipe as mp
import os

# Ruta del directorio
dataPath = "./Dataset_Train"
name = "Sin_Mascarilla" #Nombre del nuevo directorio
directory = dataPath + "/" + name


if not os.path.exists(directory):
    os.makedirs(directory)

mp_face_detection = mp.solutions.face_detection    #Crea modelo para detección de rostros en la imagen

count = 0      #Número de frames guardados

cap = cv2.VideoCapture(0)      # Captura de video desde la webcam

#Detección de rostro en la imagen captura
with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:

     while True:

          ret, frame = cap.read()    # Lectura de la imagen capturada

          if ret == False: break

          height, width, _ = frame.shape   #Concer el tamaño de la imagen (Ancho y Alto)

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    #Transformación de la imagen de BGR a RGB
          results = face_detection.process(frame_rgb)           # Detección del rostro en la imagen RGB

          # Cuando se detecta un rostro en la imagen

          if results.detections is not None:

               for detection in results.detections:
                    # Se obtienen las coordenadas del recuadro que encierra el rostro de las coordenadas relativas
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)   # X del Punto de partida del recuadro
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)  # Y del punto de partida del recuadro
                    w = int(detection.location_data.relative_bounding_box.width * width)     #Ancho del rostro
                    h = int(detection.location_data.relative_bounding_box.height * height)   #Alto del rostro
                   
                    #Para el caso cuando las coordenas están fuera de la imagen
                    if xmin < 0 and ymin < 0:
                         continue
                    
                    # Se guarda la información del rostro, desde la imagen capturada
                    face_image = frame[ymin : ymin + h, xmin : xmin + w]

                    #Se redimensiona la imagen y se interpola con el método cúbico
                    face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)

                    #Guardar la imagen con nombre face_#.jpg
                    cv2.imwrite(directory + "/face_{}.jpg".format(count), face_image)
                    count += 1
                    
          #Se muestra la imagen capturada
          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1)
          if k == 27 or count >= 300:
               break
cap.release()
cv2.destroyAllWindows()


