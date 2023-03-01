#Seguimiento
import math

#Se crea una clase que es el rastreador
class Rastreador:
    #Inicializacion de variables
    def __init__(self):
        #Se almacena las posiciones de los objetos
        self.centro_puntos = {}
        #Almacenar las posiciones centrales de los objetos
        self.id_count = 1

    def rastreo(self, objetos):
        #Almacenamos los objetos identificados
        objetos_id = []

        #Obtenemos el punto central del nuevo objeto
        for rect in objetos:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            #Miramos si ese objetp ya fue detectado
            objeto_det = False
            for id, pt in self.centro_puntos.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.centro_puntos[id] = (cx, cy)
                    print(self.centro_puntos)
                    objetos_id.append([x, y, w, h, id])
                    objeto_det = True
                    break

            #Si detecta un nuevo objeto le asignamos el ID a ese objeto
            if objeto_det is False:
                self.centro_puntos[self.id_count] = (cx, cy) #Almacenamos la coordenada x e y
                objetos_id.append([x, y, w, h, self.id_count]) #Agregamos el objeto con su ID
                self.id_count = self.id_count + 1 #Aumentamos el ID

        #Limpiar la lista por puntos centrales para eliminar IDS que ya no se usan
        new_center_points = {}
        for obj_bb_id in objetos_id:
            _, _, _, _, object_id = obj_bb_id
            center = self.centro_puntos[object_id]
            new_center_points[object_id] = center

        #Actualizar la lista con los ID no utilizados eliminados
        self.centro_puntos = new_center_points.copy()
        return objetos_id