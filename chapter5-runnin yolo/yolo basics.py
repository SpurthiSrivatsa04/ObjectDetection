from ultralytics import YOLO
import cv2
model=YOLO('../yolo weights/yolov8n.pt')
results=model("Images/1.png",show=True)
cv2.waitKey(0)