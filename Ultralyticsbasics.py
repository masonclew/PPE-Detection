#Thanks to DAVIDNYARKO123 for helping https://www.youtube.com/watch?v=hg4oVgNq7Do&t=408s

from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  

# predict on an image
detection_output = model.predict(source="videos/NewsClip.mp4", conf=0.25, save=True) 

# Display tensor array
print(detection_output)

#Video output should be under predict# 
#In cmd use the following to run the video from root dir: start wmplayer "path"
