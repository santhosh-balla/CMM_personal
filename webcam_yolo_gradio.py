import gradio as gr
from ultralytics import YOLO
import cv2
import os
import torch

# Load model immediately in background
model_path = "CMM-Yolo11.pt"
model = None

print(f"Attempting to load model from: {model_path}")
try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
        if torch.cuda.is_available():
            model.to('cuda')
            # Enable cudnn benchmark for optimized execution
            torch.backends.cudnn.benchmark = True
            print(f"✅ Model loaded on GPU: {model_path}")
        else:
            print(f"✅ Model loaded on CPU: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Global variables for frame skipping
# frame_count = 0
# last_results = []

def detect_objects(frame):
    global model
    
    if frame is None:
        return None
    
    # Ensure frame is writable (fixes "NumPy array marked as readonly" error)
    frame = frame.copy()
    
    if model is None:
        return frame
    
    # Resize for speed (standard YOLO size)
    # Reducing input resolution significantly improves speed
    target_width = 640
    height, width = frame.shape[:2]
    scale = target_width / width
    
    # Use a separate resized frame for inference, keep original 'frame' for display
    if scale < 1:
        inference_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        inference_frame = frame
        scale = 1.0
    
    try:
        # Convert to BGR for YOLO
        frame_bgr = cv2.cvtColor(inference_frame, cv2.COLOR_RGB2BGR)
        
        # Run inference
        # verbose=False prevents console spam
        # half=True enables FP16 inference on GPU (faster)
        results = model(frame_bgr, conf=0.25, verbose=False, half=True)
        
        # Draw boxes directly on the frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Scale coordinates back to original frame size
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw solid outline (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    except Exception as e:
        print(f"Error during inference: {e}")

    return frame

with gr.Blocks(title="YOLO Live Detection") as demo:
    gr.Markdown("# Live Object Detection (High Performance)")
    
    # Single component for input and output to create an "overlay" feel
    img_component = gr.Image(
        sources=["webcam"], 
        type="numpy", 
        streaming=True, 
        label="Live Detection",
        interactive=True
    )
    
    img_component.stream(detect_objects, inputs=img_component, outputs=img_component)

if __name__ == "__main__":
    demo.launch()
