import cv2
import mediapipe as mp
import numpy as np
import os
from deepface import DeepFace  # New emotion engine

# ------------------------------
# Initialize Mediapipe Modules
# ------------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

saved_background = None


# ------------------------------
# Capture background (optional)
# ------------------------------
def capture_background(frame):
    global saved_background
    saved_background = frame.copy()
    print("✅ Background captured successfully!")


# ------------------------------
# Apply Filters
# ------------------------------
def apply_filter(frame, filter_type):
    global saved_background

    # 1️⃣ Grayscale
    if filter_type == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Edge Detection
    elif filter_type == "edges":
        return cv2.Canny(frame, 100, 200)

    # 3️⃣ Smart Background Blur
    elif filter_type == "blur":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentor.process(rgb_frame)
        mask = results.segmentation_mask
        blurred = cv2.GaussianBlur(frame, (55, 55), 0)
        condition = np.stack((mask,) * 3, axis=-1) > 0.6
        return np.where(condition, frame, blurred)

    # 4️⃣ Cartoon Effect
    elif filter_type == "cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    # 5️⃣ Mediapipe Background Blur
    elif filter_type == "background_blur":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmentor.process(rgb)
        mask = results.segmentation_mask
        blurred_bg = cv2.GaussianBlur(frame, (55, 55), 0)
        condition = np.stack((mask,) * 3, axis=-1) > 0.6
        output = np.where(condition, frame, blurred_bg)
        return output

    # 6️⃣ Emoji Overlay
    elif filter_type == "emoji_face":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        emoji_path = "assets/icons/emoji.png"
        if not os.path.exists(emoji_path):
            cv2.putText(frame, "⚠️ emoji.png not found in assets/icons/", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)

                expand_x = int(box_w * 0.15)
                expand_y_top = int(box_h * 0.25)
                expand_y_bottom = int(box_h * 0.15)

                x1 = max(0, x1 - expand_x)
                y1 = max(0, y1 - expand_y_top)
                x2 = min(w, x1 + box_w + expand_x * 2)
                y2 = min(h, y1 + box_h + expand_y_top + expand_y_bottom)

                face_w, face_h = x2 - x1, y2 - y1
                emoji_resized = cv2.resize(emoji, (face_w, face_h))

                if emoji_resized.shape[2] == 4:
                    emoji_rgb = emoji_resized[:, :, :3]
                    alpha = emoji_resized[:, :, 3] / 255.0
                else:
                    emoji_rgb = emoji_resized
                    alpha = np.ones(emoji_rgb.shape[:2])

                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        emoji_rgb[:, :, c] * alpha
                        + frame[y1:y2, x1:x2, c] * (1 - alpha)
                    )

        return frame

    # 7️⃣ Expression Detector (AI via DeepFace)
        # 7️⃣ Expression Detector (AI via DeepFace)
    elif filter_type == "expression_detector":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if not results.detections:
            return frame

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                emotion_name = analysis[0]['dominant_emotion']
                confidence = analysis[0]['emotion'][emotion_name]
            except Exception:
                continue

            # Always display text
            cv2.putText(frame, f"{emotion_name.title()} ({confidence:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Try to load emotion-specific emoji
            emotion_file = f"assets/icons/{emotion_name.lower()}.png"
            fallback_file = "assets/icons/emoji.png"

            # Use fallback if emotion file doesn't exist
            if os.path.exists(emotion_file):
                emoji_path = emotion_file
            elif os.path.exists(fallback_file):
                emoji_path = fallback_file
            else:
                continue  # no emoji at all

            emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
            if emoji is None:
                continue

            emoji_resized = cv2.resize(emoji, (x2 - x1, y2 - y1))

            if emoji_resized.shape[2] == 4:
                emoji_rgb = emoji_resized[:, :, :3]
                alpha = emoji_resized[:, :, 3] / 255.0
            else:
                emoji_rgb = emoji_resized
                alpha = np.ones(emoji_rgb.shape[:2])

            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    emoji_rgb[:, :, c] * alpha
                    + frame[y1:y2, x1:x2, c] * (1 - alpha)
                )

        return frame


    # Default (no filter)
    else:
        return frame
