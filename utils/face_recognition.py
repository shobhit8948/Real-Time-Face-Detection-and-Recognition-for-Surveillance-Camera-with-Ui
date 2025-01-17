# this is face_recognition.py file
import cv2
import dlib
import numpy as np
from keras_facenet import FaceNet
import sqlite3
from datetime import datetime
from utils.database import log_entry, refine_embedding, add_person, get_all_persons_embedding

# Initialize FaceNet embedder
embedder = FaceNet()

# Initialize dlib face detector and 81-landmark shape predictor
shape_predictor_path = "models/shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Function to check if an image is blurry
def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    except Exception as e:
        print(f"Error checking blurriness: {e}")
        return True

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)

def get_face_embedding(image):
    """Extract the face embedding using FaceNet."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    embeddings = embedder.embeddings([image_rgb])
    return np.array(embeddings[0])

def align_face(image, rect):
    """Align the face using landmarks."""
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

    # Ensure the bounding box coordinates are within the image bounds
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        print("Face bounding box is out of image bounds, skipping.")
        return None

    cropped_face = image[y:y+h, x:x+w]
    
    # Check if cropped face is valid
    if cropped_face.size == 0:
        print("Cropped face is empty, skipping.")
        return None

    resized_face = cv2.resize(cropped_face, (160, 160))
    return resized_face

def recognize_and_log(image, save_path, cooldown_tracker, cooldown_period=20):
    
    if is_image_blurry(image):
        print("Detected blurry image, skipping.")
        return None

    face_embedding = get_face_embedding(image)
    all_persons = get_all_persons_embedding()

    threshold = 0.7  # Similarity threshold for recognition
    now = datetime.now()

    for person_id, stored_embedding in all_persons:
        similarity = cosine_similarity(face_embedding, stored_embedding)
        if similarity > threshold:
            # Check cooldown
            if person_id in cooldown_tracker and (now - cooldown_tracker[person_id]).total_seconds() < cooldown_period:
                print(f"Person {person_id} detected recently. Skipping logging.")
                return person_id

            # Refine embedding and log the entry
            refine_embedding(person_id, face_embedding)
            captured_face_path = f"{save_path}/person_{person_id}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(captured_face_path, image)
            log_entry(person_id, captured_face_path)
            cooldown_tracker[person_id] = now
            print(f"Recognized person_id: {person_id}, entry logged.")
            return person_id

    # Register new person
    captured_face_path = f"{save_path}/new_person_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(captured_face_path, image)
    person_id = add_person(face_embedding, captured_face_path)
    log_entry(person_id, captured_face_path)
    cooldown_tracker[person_id] = now
    print(f"New person registered with person_id: {person_id}, entry logged.")
    return person_id