import cv2 as cv
import numpy as np
from mtcnn import MTCNN


def detect_faces_haar(img, face_cascade):
    '''Detect faces in img using haar cascade 
    and return an array of bounding boxes.
    Bounding boxes on format [x, y, w, h]
    '''
    # TODO: doesn't work, mebbe remove completely
    img_copy = img.copy()
    img_gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=0.1, minNeighbors=5)
    return faces


def detect_faces_mtcnn(img, detector):
    dets = detector.detect_faces(img)
    faces = np.zeros(shape = (len(dets), 5))
    for i in range(0, len(dets)):
        x1, y1, w, h = dets[i]['box']
        faces[i,:4] = (x1, y1, x1+w, y1+h)
        faces[i,4] = dets[i]['confidence']
    return faces


def detect_faces_ssd(img, net):
    '''Detect faces in img using single shot detector (SSD) 
    and return an array of bounding boxes.
    Bounding boxes on format [x1, y1, x2, y2, score]
    '''
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # blob is to be passed through network.
    net.setInput(blob)
    detections = net.forward()
    (h, w) = img.shape[:2]
    faces = np.zeros(shape=(detections.shape[2], 5))
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            bb = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # box to image scale
            (startX, startY, endX, endY) = bb.astype("int")
            bw = endX - startX
            bh = endY - startY
            faces[i,:] = np.array((startX, startY, endX, endY, confidence))
    # Debug
    # print(faces[list(map(lambda x : np.sum(x) > 0, faces))])
    # print(faces[list(map(lambda x : np.sum(x) > 0, faces))].shape)
    return faces[list(map(lambda x : np.sum(x) > 0, faces))]


def detect_faces_yolo(img, net):
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    (height, width) = img.shape[:2]
    layerOutputs = net.forward(ln) # try without ln-argument; only need boxes...
    faces = [] 
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, W, H) = box.astype('int')

                startX = int(centerX - W / 2)
                startY = int(centerY - H / 2)
                endX = int(centerX + W / 2)
                endY = int(centerY + H / 2)
                confidences.append(float(confidence))
                faces.append([startX, startY, endX, endY, confidence])

    idxs = cv.dnn.NMSBoxes(faces, confidences, 0.5, 0.3)
    ret = np.array(faces)[idxs][:,0,:] if len(idxs) > 0 else np.array([])
    return ret


def select(method):
    if method == 'ssd':
        detector = cv.dnn.readNetFromCaffe("pre-trained-models/caffe_ssd/deploy.prototxt.txt",
            "pre-trained-models/caffe_ssd/res10_300x300_ssd_iter_140000.caffemodel")
        func = detect_faces_ssd
    if method == 'mtcnn':
        detector = MTCNN()
        func = detect_faces_mtcnn
    if method == 'yolo':
        labelsPath = "pre-trained-models/yolo-face/coco.names"
        labels = open(labelsPath).read().strip().split("\n")
        weights = "pre-trained-models/yolo-face/yolov3.weights"
        config = "pre-trained-models/yolo-face/yolov3.cfg"
        detector = cv.dnn.readNetFromDarknet(config, weights)
        func = detect_faces_yolo
    if method == 'haar':
        data_path = '/home/kaj/Documents/Uni/Exjobb/Code/venvs/py3.7/lib/python3.7/site-packages/cv2/data'
        detector = cv.CascadeClassifier(data_path + '/haarcascade_frontalface_alt.xml')
        func = detect_faces_haar

    return detector, func
