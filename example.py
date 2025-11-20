from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8n model (trained on COCO)
model = YOLO("CMM-Yolo11.pt")

# Define which class(es) you want to detect
target_classes = ["cup"]  # you can also do ["book", "laptop"] etc.

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run YOLO inference
    results = model(frame)

    # Get names dict (id → label)
    names = model.names

    # Copy frame for annotation
    annotated_frame = frame.copy()

    # Loop over detections and draw only target classes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])

        if label in target_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    cv2.imshow("Filtered YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

results = model(frame)


print(results)

