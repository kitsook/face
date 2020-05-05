import cv2

from train import train_recognizer
from detect import detect_faces, level_face
import config


def recognize_face(recognizer, image, face_cascade, eye_cascade, face_size, threshold):
    found_faces = []

    gray, faces = detect_faces(image, face_cascade, eye_cascade, return_gray=1)

    # If faces are found, try to recognize them
    for ((x, y, w, h), eyedim)  in faces:
        label, distance = recognizer.predict(cv2.resize(level_face(gray, ((x, y, w, h), eyedim)), face_size))
        if distance < threshold:
            found_faces.append((label, distance, (x, y, w, h)))

    return found_faces


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
    # no need to detect eyes location
    # eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)
    eye_cascade = None
    face_size = config.DEFAULT_FACE_SIZE
    threshold = 500

    recognizer = train_recognizer('imgdb', face_size, show_faces=True)

    cv2.namedWindow("camera", 1)
    capture = cv2.VideoCapture(0)

    while True:
        retval, img = capture.read()
        if retval:
            for (label, distance, (x, y, w, h)) in recognize_face(recognizer, img, face_cascade, eye_cascade, face_size, threshold):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "{} (d={})".format(recognizer.getLabelInfo(label), int(distance)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)

            cv2.imshow("camera", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
