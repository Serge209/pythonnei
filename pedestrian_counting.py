#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import cv2

# Object detection imports
from utils import backbone
from api import object_counting_api

#input_video = "./input_images_and_videos/york.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
roi = 385 # roi line position
deviation = 10 # the constant that represents the object counting area
custom_object_name = 'Pedestrian'
#object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation, custom_object_name) # counting all the objects


cap = cv2.VideoCapture(0)

while (True):
    frame = cap.read()
    #cv2.imshow('Video', frame)

    object_counting_api.cumulative_object_counting_x_axis(frame, detection_graph, category_index,
                                                          is_color_recognition_enabled, roi, deviation,
                                                          custom_object_name)  # counting all the objects

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()