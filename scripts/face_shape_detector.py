"""
Real-Time Face Shape Detection using Computer Vision
====================================================
Detects face shape in real-time using dlib facial landmarks and KNN classification.

Requirements:
    - OpenCV (cv2)
    - dlib
    - imutils
    - scikit-learn
    - Pre-trained models:
        * shape_predictor_68_face_landmarks.dat (dlib facial landmarks)
        * KNN_Model.sav (trained KNN classifier)

Author: Ashen Perera
Date: 2024
"""

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

# For scikit-learn >= 0.23, use joblib directly
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib


# Configuration
SHAPE_PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
KNN_MODEL_PATH = 'models/KNN_Model.sav'

# Face shape labels
FACE_SHAPE_LABELS = {
    0: 'Diamond',
    1: 'Oblong',
    2: 'Oval',
    3: 'Round',
    4: 'Square',
    5: 'Triangle'
}


def load_models():
    """
    Load face detection and classification models.
    
    Returns:
        tuple: (face_detector, landmark_detector, classifier)
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    try:
        # Initialize dlib's face detector
        face_detector = dlib.get_frontal_face_detector()
        
        # Load facial landmark predictor
        landmark_detector = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        
        # Load pre-trained KNN classifier
        classifier = joblib.load(KNN_MODEL_PATH)
        
        return face_detector, landmark_detector, classifier
    
    except FileNotFoundError as e:
        print(f"\nError: Required model file not found!")
        print(f"Details: {e}")
        print("\nPlease ensure the following files are in the 'models/' directory:")
        print(f"  1. {SHAPE_PREDICTOR_PATH}")
        print(f"  2. {KNN_MODEL_PATH}")
        print("\nDownload instructions:")
        print("  - shape_predictor_68_face_landmarks.dat:")
        print("    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("  - KNN_Model.sav: Train your own or obtain from project source")
        raise


def calculate_face_features(points):
    """
    Calculate facial features from landmark points for classification.
    
    Args:
        points (ndarray): 68 facial landmark points
    
    Returns:
        list: Normalized feature distances [d1, d2, d3, d4, d5]
    """
    # Extract relevant points (indices 2-9 represent jaw line)
    my_points = points[2:9, 0]
    
    # Calculate distances from point 6 to other points
    D1 = my_points[6] - my_points[0]
    D2 = my_points[6] - my_points[1]
    D3 = my_points[6] - my_points[2]
    D4 = my_points[6] - my_points[3]
    D5 = my_points[6] - my_points[4]
    D6 = my_points[6] - my_points[5]
    
    # Normalize distances relative to D1
    d1 = (D2 / float(D1)) * 100 if D1 != 0 else 0
    d2 = (D3 / float(D1)) * 100 if D1 != 0 else 0
    d3 = (D4 / float(D1)) * 100 if D1 != 0 else 0
    d4 = (D5 / float(D1)) * 100 if D1 != 0 else 0
    d5 = (D6 / float(D1)) * 100 if D1 != 0 else 0
    
    return [d1, d2, d3, d4, d5]


def predict_face_shape(points, classifier, img):
    """
    Predict face shape and display result on image.
    
    Args:
        points (ndarray): 68 facial landmark points
        classifier: Trained KNN classifier
        img (ndarray): Image to annotate
    
    Returns:
        str: Predicted face shape label
    """
    # Calculate features
    features = calculate_face_features(points)
    
    # Predict face shape
    result = classifier.predict([features])
    label_id = result[0]
    face_shape = FACE_SHAPE_LABELS.get(label_id, 'Unknown')
    
    # Display result on image
    text = f'FACE TYPE: {face_shape}'
    cv2.putText(img, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.3, (255, 255, 255), 2)
    
    return face_shape


def draw_landmarks(img, points):
    """
    Draw facial landmarks on the image.
    
    Args:
        img (ndarray): Image to draw on
        points (ndarray): 68 facial landmark points
    """
    for i, point in enumerate(points, start=1):
        center = (point[0], point[1])
        
        # Draw landmark point
        cv2.circle(img, center, 2, (0, 255, 0), -1)
        
        # Draw landmark number
        cv2.putText(img, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.3, (0, 255, 255), 1)


def main():
    """Main execution function for real-time face shape detection."""
    print("=" * 70)
    print("Real-Time Face Shape Detection")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    try:
        face_detector, landmark_detector, classifier = load_models()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("✗ Error: Could not access camera")
        return
    
    print("✓ Camera initialized successfully!")
    print("\nInstructions:")
    print("  - Position your face in front of the camera")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("\nStarting detection...")
    
    frame_count = 0
    
    while True:
        # Capture frame
        ret, img = camera.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector(gray)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Add green header bar
        img[0:50, 0:width] = [0, 255, 0]
        
        try:
            if len(faces) > 0:
                # Process first detected face
                face_rect = faces[0]
                
                # Get facial landmarks
                landmarks = landmark_detector(gray, face_rect)
                points = face_utils.shape_to_np(landmarks)
                
                # Draw landmarks
                draw_landmarks(img, points)
                
                # Predict and display face shape
                face_shape = predict_face_shape(points, classifier, img)
        
        except Exception as e:
            # Display error message on image
            error_text = f"Detection Error: {str(e)[:30]}"
            cv2.putText(img, error_text, (40, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Face Shape Detection - Press Q to Quit', img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f'face_shape_screenshot_{frame_count}.jpg'
            cv2.imwrite(filename, img)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("Detection stopped. Thank you!")
    print("=" * 70)


if __name__ == "__main__":
    main()
