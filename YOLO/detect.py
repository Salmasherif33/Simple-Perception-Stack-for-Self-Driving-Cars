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
def detect(img,net,output_layers):
    scale = 0.00392
    width, height = img.shape[1], img.shape[0]
    blob = cv2.dnn.blobFromImage(img, scale,(416, 416) , (0, 0, 0), crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 10)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids ,boxes , confidences

def vis(img,class_ids, boxes, confidences,classes,colors):
    # eliminate redundant overlapping boxes with conf < 0.5
    confThreshold, nmsThreshold = 0.5, 0.4
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    bbox = []
    labels = []

    for j in indexes:
        x, y, w, h = boxes[j]
        bbox.append([int(x), int(y), int(x+w), int(y+h)])
        labels.append(str(classes[class_ids[j]]))   

    return bbox,labels



def detect(img,net,output_layers):
    scale = 0.00392
    width, height = img.shape[1], img.shape[0]
    blob = cv2.dnn.blobFromImage(img, scale,(416, 416) , (0, 0, 0), crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 10)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids ,boxes , confidences

def vis(img,class_ids, boxes, confidences,classes,colors):
    # eliminate redundant overlapping boxes with conf < 0.5
    confThreshold, nmsThreshold = 0.5, 0.4
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    bbox = []
    labels = []

    for j in indexes:
        x, y, w, h = boxes[j]
        bbox.append([int(x), int(y), int(x+w), int(y+h)])
        labels.append(str(classes[class_ids[j]]))   
    for i, label in enumerate(labels):

        color = colors[i]

        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)

        cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
