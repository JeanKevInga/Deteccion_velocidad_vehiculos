#Importar librerias
import cv2
import numpy as np
from Seguidor import *
import time

#Se crea un objeto de seguimiento
seguimiento = Rastreador()

#Realizamos la lectura del video
cap = cv2.VideoCapture("Carros.mp4")

#Vamos a realizar una deteccion de objetos con camara estable
deteccion = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=100) #Se extrae los objetos en movimiento de una camara estable

#Listas para tiempos
carI = {}
carO = {}
prueba = {}

while True:
    #Lectura de la VideoCaptura
    ret, frame  = cap.read()

    #Extraemos el ancho y el alto de los fotogramas
    height = frame.shape[0]
    width = frame.shape[1]

    #Creamos una mascara
    mask = np.zeros((height, width), dtype=np.uint8)

    #Elegimos una zona de interes
    #Seleccionamos los Puntos
    pts = np.array([[[860, 360], [1037, 370], [942, 849], [54, 786]]])
    #pts = np.array([[[860,360], [1037,370], [942,849], [54,786]]])

    #Creamos los polígonos con los puntos
    cv2.fillPoly(mask, pts, 255)

    #Eliminamos lo que este fuera de los puntos
    zona = cv2.bitwise_and(frame, frame, mask=mask)

    #Mostramos con lineas la zona de interes ROIs
    areag = [(860, 360), (1037, 370), (939, 872), (49, 792)]
    area3 = [(860, 360), (1037, 370), (1017, 477), (677, 457)]
    area1 = [(501, 550), (994, 580), (941, 851), (59, 785)]
    area2 = [(677, 457), (1017, 477), (994, 580), (501, 550)]

    #Dibujamos
    #Area general
    cv2.polylines(frame, [np.array(areag, np.int32)], True, (255, 255, 0), 2)
    #Area3
    cv2.polylines(frame, [np.array(area3, np.int32)], True, (0, 130, 255), 1)
    #Area 2
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 1)
    #Area 1
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 130, 255), 1)

    #Creamos una mascara
    mascara = deteccion.apply(zona)

    #Aplicamos suavizado
    filtro = cv2.GaussianBlur(mascara, (11, 11), 0)

    #Umbral de binarizacion
    _, umbral = cv2.threshold(filtro, 50, 255, cv2.THRESH_BINARY)

    #Dilatamos los pixeles
    dila = cv2.dilate(umbral, np.ones((3, 3)))

    #Creamos un Kernel (mascara)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #Aplicamos el kernel para juntar los pixeles dispersos
    cerrar = cv2.morphologyEx(dila, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(cerrar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = [] #Lista donde vamos a almacenar la informacion

    #Dibujamos todos los contornos en frame
    for cont in contornos:

        #Eliminamos los contornos pequeños
        area = cv2.contourArea(cont)
        if area > 1800:

            #cv2.drawContours(zona, [cont], -1, (255, 255, 0), 2)
            x, y, ancho, alto = cv2.boundingRect(cont)

            #Dibujamos el rectangulo
            #cv2.rectangle(zona, (x , y), (x + ancho, y + alto, (255, 255, 0), 3)

            #Almacenamos la informacion de las detecciones
            detecciones.append([x, y, ancho, alto])

    #Seguimiento de los objetos
    info_id = seguimiento.rastreo(detecciones)

    for inf in info_id:

        #Extraemos coordenadas
        x, y, ancho, alto, id = inf

        #Dibujamos el rectangulo
        cv2.rectangle(frame, (x, y - 10), (x + ancho, y + alto), (0, 0, 255), 2) #Dibujamos el rectangulo

        #Extraemos el centro
        cx = int(x + ancho / 2)
        cy = int(y + alto / 2)

        #Area de influencia
        a2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)

        #Si esta en el centro de la mitad
        if a2 >= 0:
            #Tomamos el tiempo en el que el carro entro
            carI[id] = time.process_time()

        if id in carI:
            #Mostramos el centro
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            #Preguntamos si entra al area 3
            a3 = cv2.pointPolygonTest(np.array(area3, np.int32), (cx, cy), False)

            #Si esta en el area
            if a3 >= 0:
                #Tomamos el tiempo
                tiempo = time.process_time() - carI[id]

                #Corregimos el error de tiempo
                if tiempo % 1 == 0:
                    tiempo = tiempo + 0.323

                if tiempo % 1 != 0:
                    tiempo = tiempo + 1.016

                if id not in carO:
                    #Almacenamos la info
                    carO[id] = tiempo

                if id in carO:
                    tiempo = carO[id]

                    vel = 14.3 / carO[id]
                    vel = vel * 3.6

                #Mostramos el numero
                cv2.rectangle(frame, (x, y - 10), (x + 100, y - 50), (0, 0, 255), -1)
                cv2.putText(frame, str(int(vel)) + "KM / H", (x, y - 35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        #Mostramos el numero
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    #Mostramos los frames
    cv2.imshow("Carretera", frame)

    # Mostramos la mascara
    #cv2.imshow("Mascara", zona)

    # Mostramos la mascara
    # cv2.imshow("Mascara", filtro)

    # Mostramos la mascara
    # cv2.imshow("Mascara", umbral)

    #Mostramos la mascara
    #cv2.imshow("Mascara", dila)

    # Mostramos la mascara
    # cv2.imshow("Mascara", cerrar)

    key = cv2.waitKey(5)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
