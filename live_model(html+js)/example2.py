from ultralytics import YOLO

model = YOLO("CMM-Yolo11.pt")
model.export(format="onnx")

print(model.model.names)