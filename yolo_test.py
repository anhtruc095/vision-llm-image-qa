from ultralytics import YOLO
import numpy as np

# load model (auto download)
model = YOLO("yolov8n.pt")

# predict
results = model("test.jpeg", save=True)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        print(model.names[cls], round(conf, 2))