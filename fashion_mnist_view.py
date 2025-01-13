import streamlit as st
import numpy as np
import random
from tensorflow.keras.models import load_model
from PIL import Image
from src.load_data import cargar_datos

modelo = load_model('results/modelo_entrenado.keras')

def cargar_datos_prueba():
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_datos()
    return x_prueba, y_prueba

clases = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_prueba, y_prueba = cargar_datos_prueba()

# Cargar el CSS desde el archivo
with open("styles/styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Clasificador Fashion MNIST")

col = st.columns(3)[1]

with col:
    mostrar_imagen = st.button("Mostrar Imagen Aleatoria")

if mostrar_imagen:
    idx = random.randint(0, len(x_prueba) - 1)
    img = x_prueba[idx]
    etiqueta = y_prueba[idx]
    
    img = img.reshape(28, 28)
    img_input = img.reshape(1, 28, 28, 1)
    
    prediccion = modelo.predict(img_input)
    clase_predicha = np.argmax(prediccion, axis=1)[0]
    
    with col:
        st.write("")
        st.image(Image.fromarray((img * 255).astype(np.uint8)), width=500)
    
    with col:
        st.write(f"Predicción: {clases[clase_predicha]}")
        st.write(f"Real: {clases[etiqueta]}")

footer_html = """<div class="footer">
  <p>© 2025 Desarrollado By Lautaro Avila</p>
</div>"""

st.markdown(footer_html, unsafe_allow_html=True)
