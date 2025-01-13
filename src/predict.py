import numpy as np
import pandas as pd

def predecir_y_guardar(modelo, x_prueba, archivo_salida='predicciones.csv'):
    predicciones = modelo.predict(x_prueba)
    clases_predichas = np.argmax(predicciones, axis=1)
    df = pd.DataFrame({
        'id': np.arange(60001, 60001 + len(clases_predichas)),
        'label': clases_predichas
    })
    df.to_csv(archivo_salida, index=False)
    print(f"Predicciones guardadas en {archivo_salida}")
