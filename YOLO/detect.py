import cv2
import numpy as np
import matplotlib.pyplot as plt

yolov3_cfg_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg'
yolov3_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'

def load(weights_path , cfg_path ,coco_path):
    weights_total = weights_path + "yolov3.weights"
    cfg_total = cfg_path +  "yolov3.cfg"
    coco_total = coco_path + "coco.names"

    net = cv2.dnn.readNet(weights_total, cfg_total)
    classes = []
    with open(coco_total , "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # classes = ["person", "car", "bus", "truck", "motorbike", "bicycle"]
    # classes = ["person", "car",]
    layer_names = net.getLayerNames()
    output_layers =  [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net ,classes ,output_layers,colors 




