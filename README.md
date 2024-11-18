# Face Detection with Masks
Detección de rostros con cubrebocas 

## Descripción
El objetivo es determinar la región de la imagen  donde se encuentra un rostro y,  evaluar si,  en dicha región, hay píxeles de tapabocas o no.

## Dataset
El dataset esta compuesto por dos directorios donde se encuentran 1512 imagenes, donde 756 de ellas contienen rostros de personas que tienen cubrebocan, las demás 756 imágenes contienen rostros de personas que no tienen cubre bocas.

Parte del dataset está creado a partir de script "Dataset.py", el cual toma cada fotograma del video capturado por la cámara y lo modifica de tal forma que cumpla con las dimensiones y características para el entrenamiento del modelo empleado.

El directorio "train" contiene el script "train.py" que se empleó para entrenar el modelo de detección de rostros, este genera un archivo llamado "face_mask_model.xml" que contiene los parámetros del modelo de detección de rostros.

## Funcionamiento

Se debe ejecutar el script "main.py" el cual activará la cámara de video del computador, del cual, se analiza cada fotograma con el modelo entrenado para detectar el rostro de la persona y si este tiene o no cubrebocas.

## Autores

Felipe Duarte
Juan José Viana

## Bibliografía
[1] Real time Face Detection and Optimal Face Mapping for Online Classes, MCP Archana, CK Nitish1 and Sandhya Harikumar, Published under licence by IOP Publishing Ltd, Journal of Physics: Conference Series, Volume 2161, 1st International Conference on Artificial Intelligence, Computational Electronics and Communication System (AICECS 2021) 28-30 October 2021, Manipal, India

[2]BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs, Valentin Bazarevsky, Yury Kartynnik, Andrey Vakunov, Karthik Raveendran, Matthias Grundmann, Google Research, 1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA

[3] L. Anillo, M. Mejia, M. Melendez*, F. Ruiz & S. Moreno-Trillos, “Detección de tapabocas en imágenes para la prevención del COVID-19 a través de redes neuronales”, Investigación y Desarrollo en TIC, vol. 12, no. 2, pp. 1 – 12., 2021.


