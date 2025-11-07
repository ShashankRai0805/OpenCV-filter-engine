from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="assets/models/yolov8n.pt"):
        print("🔍 Loading YOLO model...")
        self.model = YOLO(model_path)
        print("✅ YOLO model loaded successfully!")

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame