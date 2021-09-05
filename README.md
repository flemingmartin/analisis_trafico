# Análisis de tráfico utilizando redes neuronales para reconocimiento de vehículos y peatones.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wT50-X1Bj7p1Ncxf5yga0QcdYJo2KSjl?usp=sharing)

Trabajo de investigación en el laboratorio LEICI que consiste en la utilización de algoritmos de reconocimiento, detección y conteo de objetos con el objetivo de controlar de manera automática e inteligente las luces de un semáforo. Para esto, se utilizó YOLOv4 para las detecciones de los objetos, y el algoritmo DeepSort para detectar unívocamente estos objetos con el fin de tener un rastro del recorrido de los mismos. Todo esto corriendo sobre el framework Tensorflow. 

Una vez se obtienen las posiciones de los objetos en la cámara, se comparan estas coordenadas con algunas regiones predefinidas por el usuario para poder así detectar en qué regiones se encuentran estos objetos, con el fin de determinar, por ejemplo, cuántos vehículos se encuentran en cada esquina. Estos datos permitirán ahora sí alimentar un sistema de control que se encargará de cambiar las luces de un semáforo si lo cree necesario.

## Resultados video de USA
<p align="center"><img src="data/demo_usa.gif" width=640px></p>


## Resultados video de Japón
<p align="center"><img src="data/demo_japon.gif" width=640px></p>

## Demo
Para probar el código, te recomendamos seguir este tutorial en <a href='https://colab.research.google.com/drive/1wT50-X1Bj7p1Ncxf5yga0QcdYJo2KSjl?usp=sharing'>Colab</a> (en desarrollo).
