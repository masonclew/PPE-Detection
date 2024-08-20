from ultralytics import YOLO
from PIL import Image
import cv2


model = YOLO("yolov8n.pt")  # pass any model type
results = model.predict(source="0", show = True)