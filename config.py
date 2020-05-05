import cv2

# Depends on your system, the OpenCV pre-trained classifiers for
# face / eyes detection could be in different folder
FACE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
EYE_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_eye.xml"

# Size of the face images used for training
DEFAULT_FACE_SIZE = (200, 200)

# Filename for storing  face recognizer result
RECOGNIZER_OUTPUT_FILE = "train_result.out"
