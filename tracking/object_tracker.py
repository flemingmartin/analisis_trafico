import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pafy
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet





'''
##### Definicion de parametros de entrada ##### #

En esta seccion se detallan los parametros de entrada configurables del 
algoritmo, entre los que se encuentran:

  -regiones_path -> Direccion del fichero que contiene las regiones 
                    delimitadas por el usuario. default: regiones.txt

  -weights       -> Direccion del archivo que contiene los pesos del modelo
                    ya transformado a un modelo de Tensorflow (para pasar de
                    un modelo YOLO a uno Tensorflow ver save_model.py)

  -size          -> Tamaño de entrada de las imagenes. Debe coincidir con el
                    tamaño asignado a la hora de transformar el modelo

  -tiny          -> Define si se trata de un modelo YOLO o un modelo YOLO Tiny

  -model         -> Define si se va a utlizar YOLOv3 o YOLOv4. default: YOLOv4

  -video         -> Dirección del video de entrada, puede ser un fichero, la
                    camara web del dispositivo enviando un 0, o bien un video
                    de youtube enviando el link del mismo.

  -output        -> Dirección del video de salida. default: None (no guarda
                    la salida)

  -output_format -> Define el formato del video de salida. default: XVID

  -iou           -> Umbral de IoU

  -score         -> Umbral de score o confianza

  -dont_show     -> Enviar este parámetro para no mostrar el video de salida
                    mientras se realiza el procesamiento

  -info          -> Muestra detalles de las detecciones a medida que realiza
                    el procesamiento
'''


flags.DEFINE_string('regiones_path', 'regiones.txt','path of regiones txt file')
flags.DEFINE_string('weights', './checkpoints/yolov4-640','path to weights file')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')



def pertenece_a_region(centro,regiones):
    '''
    Función que se encarga de determinar a qué region pertenece determinada 
    coordenada pasada como parametro.

    Entradas:
      -centro: coordenadas de entrada
      -regiones: conjunto de regiones definidas por el usuario

    Salida:
      La funcion retorna la region a la que pertenece el centro, y devuelve -1 si 
      este no pertenece a ninguna region
    '''

    for i,region in enumerate(regiones):
        if (centro[0] > region[0] and centro[0] < region[2]) and (centro[1] > region[1] and centro[1] < region[3]):
            return i
    return -1




    

def main(_argv):  
    '''
    Función que se encarga de realizar todo el procesamiento sobre el video de 
    entrada, guarda el video de resultado en la ruta ingresada como parámetro, y 
    guarda ademas la información recolectada del video, siendo esta la cantidad 
    de autos por region en funicon del tiempo, y tambien todas las coordenadas de 
    todos los autos detectados. Esto con la idea de realizar un analisis 
    estadistico posterior a la ejecucion del programa 
    '''

    # Cargar regiones del archivo
    regiones = np.loadtxt(FLAGS.regiones_path,dtype=np.float32)
    cant_regiones = len(regiones)//4
    regiones = np.reshape(regiones,(cant_regiones,4))  

    # Variable que guarda el total de autos que pasaron por cada region
    total_por_region = [0]*(cant_regiones+1)
    
    # Variable que guarda para cada objeto, las regiones que transito en los 
    # ultimos frames. Esto con el objetivo de evitar que un objeto que se 
    # encuentra en el borde de una region se cuente varias veces
    historial_regiones = []
    len_historial = 20

    # Variable que guarda todas las posiciones de todos los objetos detectados
    centros = []
    
    max_objects = 5000
    for i in range(max_objects):
        centros.append([-1])
        historial_regiones.append(deque([-2]*len_historial, maxlen=len_historial))


    # Parametros de deep sort
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 0.6
    
    # Inicializar deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    tracker = Tracker(metric)

    # Cargar la configuracion
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

 
    # Cargar el modelo
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # Comenzar captura del video
    if video_path[:4] == 'http':
        video = pafy.new(video_path)
        best = video.getbest(preftype="mp4")

        vid = cv2.VideoCapture(best.url)
    else:
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

    # Configurar el output
    out = None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if FLAGS.output:
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # Desnormalizar las coordenadas de las regiones (0,1) -> (0,width)
    for i in range(cant_regiones):
        regiones[i,0] = int(regiones[i,0]*width)
        regiones[i,1] = int(regiones[i,1]*height)
        regiones[i,2] = int(regiones[i,2]*width)
        regiones[i,3] = int(regiones[i,3]*height)

    # Variable que guarda la cantidad de objetos por region en funcion del tiempo 
    cant_por_region = []
    for i in range(cant_regiones+1):
        cant_por_region.append([0])

    
    frame_num = 0
    fps_prom = 0

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'truck', 'bus']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        colors_regiones = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,0,0)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Variable que guarda en tiempo real la cantidad de autos por region
        cant_por_region_aux = [0]*(cant_regiones+1)
        
        # Actualizar detecciones
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # Calcular el centro a partir de la caja contenedora
            centro = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

            # Calcular region
            reg = pertenece_a_region(centro,regiones)

            # Actualizar el contador de objetos por region
            cant_por_region_aux[reg] += 1;

            # Guardar coordenada en el id del auto
            centros[track.track_id].append(centro)
            if (centros[track.track_id][0] == -1):
                centros[track.track_id].pop(0)

            # Actualizar el total de objetos por region en funcion de si este 
            # objeto ya estuvo o no en esta region en los ultimos frames
            hist = historial_regiones[track.track_id]
            if reg not in hist:
              if hist[-1] != -2:
                total_por_region[hist[-1]]+=1
                total_por_region[reg]+=1
            hist.append(reg)

            # Dibujar rectangulo en la imagen del color de la region
            color = colors_regiones[reg]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(centro[0]+10, centro[1]),0, 0.75, color,2)

            # Dibujar recorrido del auto con puntos
            # for i,c in enumerate(centros[track.track_id][-30:]):
            #    cv2.circle(frame,c,int(i*5/30),color,-1)

        # Mostrar la cantidad de objetos en cada region en este instante
        for r in range(cant_regiones+1):
            frame = cv2.putText(frame,'Region '+str(r+1)+': '+str(int(cant_por_region_aux[r])),(20,(r+1)*40),cv2.FONT_HERSHEY_SIMPLEX,1,colors_regiones[r],3)
            cant_por_region[r].append(int(cant_por_region_aux[r]))
        
        # Mostrar el total de objetos en cada region
        for r in range(cant_regiones):
            frame = cv2.putText(frame,'Totales region '+str(r+1)+': '+str(int(total_por_region[r])),(width-500,(r+1)*40),cv2.FONT_HERSHEY_SIMPLEX,1,colors_regiones[r],3)

        # Calcular tiempos
        final_time = time.time() - start_time
        fps = 1.0 / final_time
        print("FPS: %.2f" % fps)
        fps_prom+=fps

        # Mostrar y guardar salida
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not FLAGS.dont_show:
            scale_percent = 60 # percent of original size
            dim = (int(result.shape[1] * scale_percent / 100), int(result.shape[0] * scale_percent / 100))
              
            # resize image
            resized = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("Output Video", resized)
        if FLAGS.output:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyAllWindows()

    with open("cant_por_region.txt", "w") as output:
        output.write(str(cant_por_region))
    with open("centros.txt", "w") as output:
        output.write(str(centros))

    print("FPS promedio: ", fps_prom/frame_num)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
