import os
from tensorflow.keras.callbacks import ModelCheckpoint

def entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, archivo_modelo='modelo_entrenado.keras'):
    if not os.path.exists('results'):
        os.makedirs('results')

    ruta_modelo = os.path.join('results', archivo_modelo)
    checkpoint = ModelCheckpoint(ruta_modelo, save_best_only=True, monitor='accuracy', verbose=1)
    resultado_entrenamiento = modelo.fit(x_entrenamiento, y_entrenamiento, epochs=30, batch_size=64, callbacks=[checkpoint])
    
    return resultado_entrenamiento