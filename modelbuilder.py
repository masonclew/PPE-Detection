from ultralytics import YOLO
# Load a model for yolov8 nano
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1)  # train the model
