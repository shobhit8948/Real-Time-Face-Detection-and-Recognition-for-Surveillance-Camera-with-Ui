import cv2
from threading import Thread
import os
from utils.face_recognition import recognize_and_log, align_face, detector

class CameraStream:
    def __init__(self, save_path="captured_faces"):
        self.cap = None  # Initialize as None
        self.running = False
        self.thread = None
        self.save_path = save_path
        self.cooldown_tracker = {}  # Track cooldown for each person
        os.makedirs(self.save_path, exist_ok=True)

    def get_processed_frame(self):
        """Return the current frame being processed."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def _initialize_camera(self):
        """Initialize the camera."""
        # self.cap = cv2.VideoCapture("rtsp://admin:msspl%40123@192.168.1.150:554/cam/realmonitor?channel=1&subtype=1", cv2.CAP_FFMPEG)  # Default to webcam
        self.cap = cv2.VideoCapture(0)  # Default to webcam

    def start(self):
        """Start the camera stream."""
        if not self.running:
            self._initialize_camera()
            self.running = True
            self.thread = Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _capture_frames(self):
        """Continuously capture frames and process faces."""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                print("Camera not initialized or unavailable.")
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())

                # Skip small faces
                min_face_size = 40
                if w < min_face_size or h < min_face_size:
                    print("Face too small, skipping.")
                    continue

                # Align face and recognize
                aligned_face = align_face(frame, face)
                if aligned_face is None:
                    continue

                person_id = recognize_and_log(aligned_face, self.save_path, self.cooldown_tracker)

                # Draw a rectangle around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID: {person_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

camera_stream = CameraStream()

def start_camera():
    """Start the camera feed."""
    camera_stream.start()

def stop_camera():
    """Stop the camera feed."""
    camera_stream.stop()
