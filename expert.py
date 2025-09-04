from ultralytics import YOLO
m = YOLO("yolov8n.pt")
m.export(format="onnx", opset=12, imgsz=416, dynamic=False)  # imgsz=320~640 가감
