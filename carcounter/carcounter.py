from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
cap=cv2.VideoCapture(0) #for webcam
cap.set(3,1280)
cap.set(4,720)
cap=cv2.VideoCapture("../videos/cars (1).mp4") #for video

model=YOLO("../yolo weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
while True:
    success,img=cap.read()
    results=model(img,stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #boundingbox
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w,h=x2-x1,y2-y1

            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            #classname
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack([detections,currentArray])
    resultsTracker=tracker.update(detections)
    for results in resultsTracker:
        x1,y1,x2,y2,Id=results
        print(results)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
