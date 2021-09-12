from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
import pathlib
from object_detection.utils import label_map_util
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-640',
                    'path to weights file')
flags.DEFINE_string('model_name', 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
                    'model name (solo para SSD)')
flags.DEFINE_string('rcnn_model_name', 'faster_rcnn_resnet101_coco_2018_01_28',
                    'model name (solo para RCNN)')
flags.DEFINE_boolean('ssd', False, 'SSD?')
flags.DEFINE_boolean('rcnn', False, 'RCNN?')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/reduced_val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model
#Función que se encarga de correr una inferencia
def run_inference_for_single_image(model, image):
    image_np = np.array(image)
    image = np.asarray(image_np)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
     
    return output_dict

def main(_argv):
    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    if (FLAGS.ssd or FLAGS.rcnn) == True:
        category_index = label_map_util.create_category_index_from_labelmap('data/mscoco_label_map.pbtxt', use_display_name=True)
        #CLASSES = category_index.name

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if (FLAGS.ssd or FLAGS.rcnn) == False:
        print("-------------------YOLOv4-------------------")
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
    else:
        if FLAGS.ssd == True:
            print("-------------------SSD-------------------")
            detection_model = load_model(FLAGS.model_name)
        else:
            print("-------------------RCNN-------------------")
            detection_model = load_model(FLAGS.rcnn_model_name)


    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(FLAGS.annotation_path, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            if num>num_lines:
                break
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]   
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            if (FLAGS.ssd or FLAGS.rcnn) == False:
                image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
            else:
                image_data = np.copy(image)
            if (FLAGS.ssd or FLAGS.rcnn) == False:
                if FLAGS.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
                else:
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
                boxes = boxes.numpy()[0]
                scores = scores.numpy()[0]
                classes = classes.numpy()[0]
                valid_detections = valid_detections.numpy()[0]
                #boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            else:
                output_dict=run_inference_for_single_image(detection_model, image_data)
                boxes = output_dict['detection_boxes']
                scores = output_dict['detection_scores']
                classes = output_dict['detection_classes']
                valid_detections = output_dict['num_detections']


            # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
            #     cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections):
                    if int(classes[i]) < 0 or int(classes[i]) > NUM_CLASS: continue
                    coor = boxes[i]
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)

                    score = scores[i]
                    class_ind = int(classes[i])
                    if (FLAGS.ssd or FLAGS.rcnn) == False:
                        class_name = CLASSES[class_ind]
                    else:
                        class_name = category_index[class_ind]['name']
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


