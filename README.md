# Fashion MNIST - Clasificación de Ropa

Este proyecto tiene como objetivo resolver el desafío de Identify the apparels (Fashion MNIST) mediante la **clasificación de imágenes**, utilizando técnicas de **aprendizaje profundo**. La tarea consiste en reconocer el tipo de prenda de ropa a partir de imágenes.

Para ello, se utiliza el conjunto de datos **Fashion MNIST**, creado por **Zalando Research**, que contiene imágenes de distintos tipos de ropa. Este dataset consta de un total de **70,000 imágenes**, de las cuales **60,000 están etiquetadas** para entrenamiento y **10,000 son imágenes de prueba** sin etiquetar.

[**Probar el modelo de clasificación de ropas**](https://lautaroav-clasificador-fashion-mnist-fashion-mnist-view-h3nyg6.streamlit.app/)

La página web carga automáticamente el conjunto de datos de prueba, predice el tipo de prenda de ropa para cada imagen y muestra si la predicción fue correcta o no

## Descripción del Problema

Más del 25% de los ingresos del comercio electrónico provienen de la venta de ropa y accesorios. Un desafío importante en la industria es la **clasificación automática** de prendas de ropa a partir de imágenes, especialmente cuando las categorías proporcionadas por las marcas pueden ser inconsistentes. Este proyecto tiene como objetivo abordar este reto mediante el uso de **visión por computadora** y **modelos de aprendizaje profundo**.

El conjunto de datos de Fashion MNIST contiene imágenes en escala de grises de **28x28 píxeles** que representan **10 categorías de ropa**. La tarea consiste en clasificar las imágenes de prueba en una de estas categorías.

Las categorías de ropa en el conjunto de datos son las siguientes:

| **Label** | **Descripción** |
| --------- | --------------- |
| 0         | T-shirt/top     |
| 1         | Trouser         |
| 2         | Pullover        |
| 3         | Dress           |
| 4         | Coat            |
| 5         | Sandal          |
| 6         | Shirt           |
| 7         | Sneaker         |
| 8         | Bag             |
| 9         | Ankle boot      |

## Evaluación

El modelo ha alcanzado una precisión del **92%** al identificar correctamente el tipo de prenda en las imágenes de prueba.
