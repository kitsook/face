import math
import cv2
import numpy as np
import config

# given an image with the dimensions of face and eyes, return the face in leveled position
def level_face(image, face):
    ((x, y, w, h), eye_dim) = face
    if len(eye_dim) != 2:
        # only rotate the face if thre are two eyes found
        return image[y:y+h, x:x+w]

    leftx = eye_dim[0][0]
    lefty = eye_dim[0][1]
    rightx = eye_dim[1][0]
    righty = eye_dim[1][1]
    if leftx > rightx:
        leftx, rightx = rightx, leftx
        lefty, righty = righty, lefty
    if lefty == righty or leftx == rightx:
        return image[y:y+h, x:x+w]

    rotDeg = math.degrees(math.atan((righty - lefty) / float(rightx - leftx)))
    if abs(rotDeg) < 20:
        rotMat = cv2.getRotationMatrix2D((leftx, lefty), rotDeg, 1)
        rotImg = cv2.warpAffine(image, rotMat, (image.shape[1], image.shape[0]))
        return rotImg[y:y+h, x:x+w]

    return image[y:y+h, x:x+w]

def detect_faces(image, face_cascade, eye_cascade=None, return_gray=True):
    scale_factor = 1.3
    min_neighbors = 3

    # Convert color input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

    # If faces are found
    result = []
    for (x, y, w, h) in faces:
        eyes = []
        if eye_cascade:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)[:2]

        result.append(((x, y, w, h), eyes))

    if return_gray:
        return gray, result
    else:
        return image, result

if __name__ == '__main__':

    cv2.namedWindow("camera", 1)
    capture = cv2.VideoCapture(0)

    # To improve performance, can specify a lower resolution. e.g. 320x240
    width = None
    height = None

    face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
    eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)


    if width is None:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height is None:
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    result = np.zeros((height, width, 3), np.uint8)

    while True:
        retval, img = capture.read()

        image, face_dim = detect_faces(img, face_cascade, eye_cascade, False)
        for ((x, y, w, h), eye_dim) in face_dim:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eye_dim:
                cv2.rectangle(image, (ex+x, ey+y), (ex+x+ew, ey+y+eh), (0, 255, 0), 2)

        cv2.imshow("camera", image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
