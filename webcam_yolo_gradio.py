import gradio as gr
from ultralytics import YOLO
import cv2
import os

# Global model variable
model = None

def load_model(file_path):
    global model
    if file_path is None:
        return "No file uploaded."
    try:
        # file_path is the path to the uploaded file
        model = YOLO(file_path)
        return f"Model loaded: {os.path.basename(file_path)}"
    except Exception as e:
        return f"Error loading model: {e}"

def detect_objects(frame):
    global model
    if frame is None:
        return None
    if model is None:
        # If no model is loaded, just return the frame
        return frame
    
    try:
        # Convert to BGR for YOLO
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model(frame_bgr, conf=0.25)
        annotated_bgr = results[0].plot()
        # Convert back to RGB for Gradio
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error during inference: {e}")
        return frame

with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Live Detection with Custom Model")
    
    with gr.Row():
        model_file = gr.File(label="Upload .pt Model File", file_types=[".pt"])
        load_status = gr.Textbox(label="Status", value="Upload a model to start.")
    
    model_file.change(load_model, inputs=model_file, outputs=load_status)
    
    with gr.Row():
        input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam")
        output_img = gr.Image(label="Detection Output")
    
    # Stream the input to the detection function
    input_img.stream(detect_objects, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()
