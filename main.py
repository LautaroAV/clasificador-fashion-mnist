from src.load_data import cargar_datos
from src.model import crear_modelo
from src.train import entrenar_modelo
from src.predict import predecir_y_guardar

if __name__ == "__main__":
    # Cargar datos
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_datos()

    # Crear modelo
    modelo = crear_modelo()

    # Entrenar modelo
    print("Entrenando el modelo...")
    entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento)

    # Evaluar modelo
    print("Evaluando en conjunto de prueba...")
    perdida_prueba, exactitud_prueba = modelo.evaluate(x_prueba, y_prueba, verbose=2)
    print(f"Exactitud en prueba: {exactitud_prueba:.2f}")

    # Predecir y guardar resultados
    predecir_y_guardar(modelo, x_prueba)
