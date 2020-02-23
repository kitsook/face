import os
from os.path import join, isdir
from PIL import Image

import cv2
import numpy as np

from detect import detect_faces, level_face
import config

def train_recognizer(db_folder, train_size=config.DEFAULT_FACE_SIZE, show_faces=False, force_train=False):
    """ Train and return face recognier.

    db_folder -- the image folder that group faces in sub-folders
    train_size -- tuple of x and y size for resizing faces found before training
    show_faces -- display images of faces found and used for training
    force_train -- force re-training even when previous training result is found
    """
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer = cv2.face.FisherFaceRecognizer_create()
    # recognizer = cv2.face.EigenFaceRecognizer_create()

    if (not force_train) and _load_recognizer(recognizer):
        return recognizer

    folders = _get_labels(db_folder)
    images = []
    labels = []

    label_count = 0
    label_map = {}

    for folder in folders:
        faces = _extract_faces(db_folder, folder, True)

        # resize all faces to same size for some recognizers
        images.extend([cv2.resize(face, train_size) for face in faces])

        labels.extend([label_count] * len(faces))
        label_map[label_count] = folder
        label_count += 1

        if show_faces:
            cv2.namedWindow("faces", 1)
            cv2.imshow("faces", _combine_faces(faces))
            print("Press any key to continue...")
            cv2.waitKey(0)

    if show_faces:
        cv2.destroyWindow("faces")

    recognizer.train(images, np.array(labels))
    for key in label_map:
        recognizer.setLabelInfo(key, label_map[key])

    _save_recognizer(recognizer)

    return recognizer

def _get_labels(a_dir):
    return [name for name in os.listdir(a_dir) if isdir(join(a_dir, name))]

def _supported_img(name):
    return name.lower().endswith('.png') or name.lower().endswith('.jpg') or name.lower().endswith('.jpeg')

def _combine_faces(faces, w=100, h=100, num_per_row=5):
    small_img = []
    row_img = []
    count = 0
    for img in faces:
        small_img.append(cv2.resize(img, (w, h)))
        count += 1
        if count % num_per_row == 0:
            count = 0
            row_img.append(np.concatenate(small_img, axis=1))
            small_img = []
    if len(small_img) > 0:
        for x in range (0, num_per_row-len(small_img)):
            small_img.append(np.zeros((h,w), np.uint8))
        row_img.append(np.concatenate(small_img, axis=1))

    return np.concatenate(row_img, axis=0)

def _extract_faces(a_dir, folder, do_level_face=False):
    faceCascade = cv2.CascadeClassifier(config.FACE_CASCADE_FILE)
    eyeCascade = cv2.CascadeClassifier(config.EYE_CASCADE_FILE)

    the_path = join(a_dir, folder)
    result = []

    for img in [f for f in os.listdir(the_path) if _supported_img(f)]:
        img_path = join(the_path, img)
        image, faces = detect_faces(cv2.imread(img_path), faceCascade, eyeCascade, True)
        if len(faces) == 0:
            print("No face found in " + img_path)
        for ((x, y, w, h), eyedim) in faces:
            if not do_level_face:
                result.append(image[y:y+h, x:x+w])
            else:
                result.append(level_face(image, ((x, y, w, h), eyedim)))
                #result.append(image[y:y+h, x:x+w])
    return result

def _save_recognizer(recognizer, filename=config.RECOGNIZER_OUTPUT_FILE):
    recognizer.save(filename)

def _load_recognizer(recognizer, filename=config.RECOGNIZER_OUTPUT_FILE):
    try:
        recognizer.read(filename)
        return True
    except (cv2.error):
        return False

if __name__ == '__main__':
    recognizer = train_recognizer('imgdb', show_faces=True, force_train=True)
