import cv2
from filters.basic_filters import apply_filter
from detection.yolo_detector import YOLODetector

# Initialize YOLO model
detector = YOLODetector("assets/models/yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
current_filter = "none"

print("\n🎥 Camera started...")
print("🔘 Controls:")
print("0 - None")
print("1 - Grayscale")
print("2 - Blur (background only)")
print("3 - Edge Detection")
print("4 - Cartoon")
print("5 - Emoji Overlay 😎")
print("6 - Expression Detector")
print("q - Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply selected filter
    frame = apply_filter(frame, current_filter)

    # Skip YOLO when emoji filter is active (to avoid slowdowns)
    if len(frame.shape) == 3 and current_filter != "emoji_face":
        frame = detector.detect_objects(frame)

    # Display output
    cv2.imshow("Smart Camera (YOLO + Filters)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('0'):
        current_filter = "none"
    elif key == ord('1'):
        current_filter = "gray"
    elif key == ord('2'):
        current_filter = "blur"
    elif key == ord('3'):
        current_filter = "edges"
    elif key == ord('4'):
        current_filter = "cartoon"
    elif key == ord('5'):
        current_filter = "emoji_face"
    elif key == ord('6'):
        current_filter = "expression_detector"


cap.release()
cv2.destroyAllWindows()
print("👋 Camera closed.")
