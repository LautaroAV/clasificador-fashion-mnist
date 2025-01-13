import gzip, struct, os
import numpy as np

RUTA_DATA = './data/'

def cargar_datos():
    ruta_imagenes_entrenamiento = os.path.join(RUTA_DATA, 'train-images-idx3-ubyte.gz')
    ruta_etiquetas_entrenamiento = os.path.join(RUTA_DATA, 'train-labels-idx1-ubyte.gz')
    ruta_imagenes_prueba = os.path.join(RUTA_DATA, 't10k-images-idx3-ubyte.gz')
    ruta_etiquetas_prueba = os.path.join(RUTA_DATA, 't10k-labels-idx1-ubyte.gz')

    def cargar_datos_comprimidos(ruta):
        with gzip.open(ruta, 'rb') as archivo:
            return archivo.read()

    def leer_datos_idx(datos):
        datos_idx = struct.unpack('>IIII', datos[:16])
        num_elementos, filas, columnas = datos_idx[1], datos_idx[2], datos_idx[3]
        return np.frombuffer(datos[16:], dtype=np.uint8).reshape(num_elementos, filas, columnas)

    def leer_etiquetas(datos):
        return np.frombuffer(datos[8:], dtype=np.uint8)

    x_entrenamiento = leer_datos_idx(cargar_datos_comprimidos(ruta_imagenes_entrenamiento)) / 255.0
    y_entrenamiento = leer_etiquetas(cargar_datos_comprimidos(ruta_etiquetas_entrenamiento))
    x_prueba = leer_datos_idx(cargar_datos_comprimidos(ruta_imagenes_prueba)) / 255.0
    y_prueba = leer_etiquetas(cargar_datos_comprimidos(ruta_etiquetas_prueba))

    x_entrenamiento = x_entrenamiento.reshape(-1, 28, 28, 1)
    x_prueba = x_prueba.reshape(-1, 28, 28, 1)

    return (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba)
