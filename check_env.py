import sys
import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO

print("✅ Python executable path:", sys.executable)
print("✅ Python version:", sys.version)
print("\nInstalled package versions:")
print("OpenCV:", cv2.__version__)
print("Mediapipe:", mp.__version__)
print("Torch:", torch.__version__)

# Test YOLO model load
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully ✅")
