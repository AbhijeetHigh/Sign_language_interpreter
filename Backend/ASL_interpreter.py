"""
Pipeline:
    Webcam
      -> MediaPipe Hand Landmarker
      -> 21 hand landmarks
      -> wrist + scale normalization
      -> trained SVC model
      -> confidence filtering
      -> prediction smoothing
      -> on-screen result

Expected files in the same folder:
    SVC_ASL_469.pkl
    hand_landmarker.task
"""

from collections import Counter, deque
from pathlib import Path
import time

import cv2 as cv
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

MODEL_PATH = Path("SVC_1069.pkl")
LANDMARKER_PATH = Path("hand_landmarker.task")

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# How often a new prediction is made.
CAPTURE_INTERVAL = 0.25

# Ignore predictions below this confidence.
MIN_PREDICTION_CONFIDENCE = 0.40

# Number of recent predictions used for majority voting.
SMOOTHING_WINDOW = 5

# MediaPipe settings.
NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.40
MIN_HAND_PRESENCE_CONFIDENCE = 0.40
MIN_HAND_TRACKING_CONFIDENCE = 0.40

WINDOW_NAME = "ASL Sign Interpreter"


# -----------------------------------------------
# LOAD MODEL
# -----------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Put SVC_ASL_469.pkl in the same folder as this script."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not have a predict() method.")

    return model


# -----------------------------------------------
# MEDIAPIPE SETUP
# -----------------------------------------------

def create_landmarker():
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {LANDMARKER_PATH}\n"
            "Download hand_landmarker.task and put it beside this script."
        )

    base_options = python.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_HAND_TRACKING_CONFIDENCE,
    )

    return vision.HandLandmarker.create_from_options(options)


# -----------------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------------

def extract_landmarks(results):
    """
    Convert the first detected hand into 63 normalized features.

    Normalization:
      1. Move the wrist (landmark 0) to the origin.
      2. Scale by the largest XY distance from the wrist.

    This matches the preprocessing used in the supplied code.
    """

    if not results.hand_landmarks:
        return None

    hand = results.hand_landmarks[0]

    if len(hand) != 21:
        return None

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32,
    )

    # Move wrist to (0, 0, 0).
    landmarks -= landmarks[0].copy()

    # Scale normalization using XY distance from wrist.
    distances = np.linalg.norm(landmarks[:, :2], axis=1)
    max_distance = float(np.max(distances))

    if max_distance <= 1e-8:
        return None

    landmarks /= max_distance

    return landmarks.flatten()


# -----------------------------------------------
# DRAWING
# -----------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame, results):
    """Draw detected hand landmarks on the camera frame."""

    if not results.hand_landmarks:
        return

    height, width = frame.shape[:2]

    for hand in results.hand_landmarks:
        points = []

        for landmark in hand:
            x = int(np.clip(landmark.x * width, 0, width - 1))
            y = int(np.clip(landmark.y * height, 0, height - 1))
            points.append((x, y))

            cv.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for start, end in HAND_CONNECTIONS:
            cv.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2,
            )


def draw_status(
    frame,
    prediction,
    confidence,
    interval,
    hand_detected,
):
    """Draw interpreter information on the screen."""

    # Top information.
    cv.putText(
        frame,
        f"Capture: {interval:.2f}s",
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    # Main prediction.
    if not hand_detected:
        text = "Show one hand"
    elif prediction is None:
        text = "Waiting..."
    else:
        text = f"SIGN: {prediction}"

    cv.putText(
        frame,
        text,
        (20, 85),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    # Confidence.
    if prediction is not None:
        cv.putText(
            frame,
            f"Confidence: {confidence:.1%}",
            (20, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    cv.putText(
        frame,
        "Q = quit | R = reset prediction",
        (20, frame.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.60,
        (220, 220, 220),
        1,
        cv.LINE_AA,
    )


# -----------------------------------------------
# PREDICTION
# -----------------------------------------------

def predict_sign(model, features):
    """
    Return (label, confidence).

    predict_proba is preferred because it lets us reject uncertain
    predictions instead of displaying every SVC guess.
    """

    features_2d = np.asarray(features, dtype=np.float32).reshape(1, -1)

    if hasattr(model, "n_features_in_"):
        expected = int(model.n_features_in_)

        if features_2d.shape[1] != expected:
            raise ValueError(
                f"Feature mismatch: model expects {expected}, "
                f"but interpreter produced {features_2d.shape[1]}."
            )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_2d)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = model.classes_[best_index]
        return str(label), confidence

    # Fallback for a classifier without predict_proba.
    label = model.predict(features_2d)[0]
    return str(label), 1.0


# -----------------------------------------------
# SMOOTHING
# -----------------------------------------------

class PredictionSmoother:
    """
    Majority-vote smoother.
    """

    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def add(self, label):
        self.history.append(label)

    def get(self):
        if not self.history:
            return None

        return Counter(self.history).most_common(1)[0][0]

    def clear(self):
        self.history.clear()


# -----------------------------------------------
# CAMERA
# -----------------------------------------------

def open_camera():
    cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}."
        )

    # Request a reasonable resolution.
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Reduce unnecessary buffering where supported.
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    return cap


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main():
    print("=" * 55)
    print("    ASL SIGN INTERPRETER")
    print("=" * 55)

    # ---- Capture interval ----
    while True:
        try:
            interval = float(
                input(
                    f"Prediction interval in seconds "
                    f"[default {CAPTURE_INTERVAL}]: "
                ) or CAPTURE_INTERVAL
            )

            if interval > 0:
                break

            print("Please enter a value greater than 0.")

        except ValueError:
            print("Please enter a valid number.")

    print(f"\nPrediction interval: {interval:.2f} seconds")
    print("Press Q to quit.")
    print("Press R to reset prediction history.\n")

    # ---- Load components ----
    print("Loading SVC model...")
    model = load_model()

    print("Loading MediaPipe...")
    landmarker = create_landmarker()

    print("Opening camera...")
    cap = open_camera()

    smoother = PredictionSmoother(SMOOTHING_WINDOW)

    current_prediction = None
    current_confidence = 0.0
    status = "Waiting..."
    last_prediction_time = 0.0

    # MediaPipe VIDEO mode needs monotonically increasing timestamps.
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read a frame from the camera.")
                break

            # Mirror webcam so movement feels natural.
            frame = cv.flip(frame, 1)

            now = time.perf_counter()

            # BGR -> RGB.
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            # Timestamp must increase for VIDEO mode.
            timestamp_ms = int((now - start_time) * 1000)

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            draw_landmarks(frame, results)

            hand_detected = bool(results.hand_landmarks)

            # Predict only at the requested interval.
            if now - last_prediction_time >= interval:
                last_prediction_time = now

                if not hand_detected:
                    current_prediction = None
                    current_confidence = 0.0
                    status = "No hand detected"
                    smoother.clear()

                else:
                    features = extract_landmarks(results)

                    if features is None:
                        current_prediction = None
                        current_confidence = 0.0
                        status = "Invalid landmarks"

                    else:
                        try:
                            label, confidence = predict_sign(
                                model,
                                features,
                            )

                            if confidence >= MIN_PREDICTION_CONFIDENCE:
                                smoother.add(label)

                                current_prediction = smoother.get()
                                current_confidence = confidence
                                status = "Prediction accepted"
                            else:
                                status = "Low confidence"

                        except Exception as exc:
                            current_prediction = None
                            current_confidence = 0.0
                            status = "Prediction error"
                            print(
                                f"Prediction error: {exc}"
                            )

            # ---- Display ----
            draw_status(
                frame,
                current_prediction,
                current_confidence,
                interval,
                hand_detected,
            )

            cv.putText(
                frame,
                status,
                (20, 150),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv.LINE_AA,
            )

            cv.imshow(WINDOW_NAME, frame)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                smoother.clear()
                current_prediction = None
                current_confidence = 0.0
                status = "Prediction reset"

    finally:
        cap.release()
        cv.destroyAllWindows()
        landmarker.close()

    print("\nInterpreter closed.")


if __name__ == "__main__":
    main()
