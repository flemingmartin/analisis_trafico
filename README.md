# Análisis de tráfico utilizando redes neuronales para reconocimiento de vehículos y peatones.


Trabajo de investigación en el laboratorio LIFIA que consiste en la utilización de algoritmos de reconocimiento, detección y conteo de objetos con el objetivo de controlar de manera automática e inteligente las luces de un semáforo. Para esto, se utilizó YOLOv4 para las detecciones de los objetos, y el algoritmo DeepSort para detectar unívocamente estos objetos con el fin de tener un rastro del recorrido de los mismos. Todo esto corriendo sobre el framework Tensorflow. 

Una vez se obtienen las posiciones de los objetos en la cámara, se comparan estas coordenadas con algunas regiones predefinidas por el usuario para poder así detectar en qué regiones se encuentran estos objetos, con el fin de determinar, por ejemplo, cuántos vehículos se encuentran en cada esquina. Estos datos permitirán ahora sí alimentar un sistema de control que se encargará de cambiar las luces de un semáforo si lo cree necesario.

## Resultados video de USA
<p align="center"><img src="assets/demo_usa.gif" width=640px></p>


## Resultados video de Japón
<p align="center"><img src="assets/demo_japon.gif" width=640px></p>

## Demo
Para probar el código, te recomendamos seguir este tutorial en <a href='#'>Colab</a> (en desarrollo).