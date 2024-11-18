##################    Librerías requeridas para el funcionamento del clasificador y de la interfaz de usuario
import cv2
import mediapipe as mp

from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()     #Modelo de reconocimiento para la clasificación de rostros
face_mask.read("./train/face_mask_model.xml")                #Lectura del modelo entrenado anteriormente con dataset

def detector_mascarilla():
    # Función para la detección y la clasificación de los rostros
    global cap

    mp_face_detection = mp.solutions.face_detection    # Modelo de detección de rostros de mediapipe
    LABELS = ["Con_mascarilla", "Sin_mascarilla"]      # Etiquetas de clasificación
    
    #Detección del rostro con mediapipe
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read() #Lee el frame de la imagen capturada

            if ret == False: break
            frame = cv2.flip(frame, 1)           # Transformación de imagen
            height, width, _ = frame.shape       #Obtiene el tamaño de la imagen (Alto y ancho)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Tranformación de BGR a RGB
            results = face_detection.process(frame_rgb)        #Detección de rostro en el frame


            if results.detections is not None:

                for detection in results.detections:
                    ###########    Resultados de la detección en la imagen capturada    ###############

                    #  Obtención de coordenadas del rostro en la imagen

                        xmin = int(detection.location_data.relative_bounding_box.xmin * width)     # X del Punto de partida del recuadro
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)    # Y del punto de partida del recuadro
                        w = int(detection.location_data.relative_bounding_box.width * width)       # Ancho del rostro
                        h = int(detection.location_data.relative_bounding_box.height * height)     # Alto del rostro
                    
                    #Para el caso cuando las coordenas están fuera de la imagen
                        if xmin < 0 and ymin < 0:
                            continue
                    
                    ###########     Procesamiento de la imagen     ###################

                    #  Se toma el recuadro de la imagen donde está el rostro
                        face_image = frame[ymin : ymin + h, xmin : xmin + w]
                    # Se hace una transformación de RGB a Gris
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    # Se redimensiona la imagen y se interpola con el método de interpolación cúbica
                        face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)
                    
                    ###########    Clasificación del rostro     #################

                    #Entrega de la imagen del rostro en Gris y redimensionada para el clasificador
                        result = face_mask.predict(face_image)
                    
                    ###########   Muestra del resultado del rostro en pantalla    ###############

                    # Se dibuja el cuadro de color correspondiente a la clasificación asignada
                        if result[1] < 150:
                            color = (0, 255, 0) if LABELS[result[0]] == "Con_mascarilla" else (0, 0, 255)   #Color y etiqueta de acuerdo al resultado
                            
                            # impresión de texto indicando si tiene o no mascarilla
                            cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                            
                            #dibujo del rectángulo que encierra el rostro
                            cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 4)
            return ret, frame


                                    #############     INTERFAZ GRÁFICA ####################

########################   Inicie la captura de video y lo muestre en la interfaz
def iniciar():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizar()

#########################   Ver de la clasificación en la interfaz gráfica
def visualizar():
    global cap

    if cap is not None:

        ret, frame = detector_mascarilla()         # Resultado de la clasificación del rostro
        if ret == True:
            frame = imutils.resize(frame, width=640)   #se redimensiona la imagen
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Se convierte a RGB la imagen
            im = Image.fromarray(frame)                    # Se crea una imagen desde el arreglo
            img = ImageTk.PhotoImage(image=im)             #Se muestra en la interfaz
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            lblVideo.image = ""
            cap.release()
            cv2.destroyAllWindows()

###########  cierre de la captura y muestra de video, y cierre de la ventana de la interfaz de usuario
def finalizar():  
    global cap
    cap.release()
    cv2.destroyAllWindows()

##################   Diseño de la interfaz gráfica con sus respectivos botones
cap = None
root = Tk()
root.title("Detección de Tapabocas")   #Título de la ventana
btnIniciar = Button(root, text="Iniciar", width=45, command=iniciar)  # Botón de Iniciar con su función resptectiva
btnIniciar.grid(column=0, row=0, padx=5, pady=5)                      # Aspecto del botón
btnFinalizar = Button(root, text="Finalizar", width=45, command= root.quit)   # Cerrar ventana cuando se apreta el botón de finalizar
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)                    # Aspecto del botón
lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)
root.mainloop()

