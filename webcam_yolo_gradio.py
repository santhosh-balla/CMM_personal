import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("CMM-Yolo11.pt")

def detect_objects(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(frame, conf=0.25)
    annotated = results[0].plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Live Webcam Feed"),
    outputs="image",
    title="YOLOv8 Live Detection",
    live=True,  # this enables continuous webcam streaming
    description="Real-time YOLO object detection"
)

demo.launch()
