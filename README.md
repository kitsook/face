#face

An experiment with OpenCV 3.x face detection and recognition.  Note that you
will need to install the corresponding `opencv_contrib` modules for the face
recognizer to work.

## Face detection

`detect.py` contains sample on running face detection on webcam attached to the
computer. The  `levelFace` function tries to rotate the image so that the
face detected is level (based on the eye positions).

## Face recognizer training

`train.py` trains the OpenCV face recognizer by extracting faces from images
provided under a given folder. Images for each individual should be
organized in corresponding sub-folders with the folder name used by face
recognizer as the labels. e.g.:

* `imgdb/Barack Obama/image1.jpg`
* `imgdb/Barack Obama/image2.jpg`
* ...
* `imgdb/Donald Trump/anotherimage.png`
* `imgdb/Donald Trump/yetanotherimage.jpg`
* ...
* `imgdb/Justin Trudeau/faces.jpg`
* ...

Note that each image can contain multiple faces of the same person.

## Face recognition

`recognize.py` puts everything together.  It demonstrates on training the face
recognizer and feeding webcam images to recognize faces found.

Note that in order to speed up the process, the training result should run once
and saved.  Subsequent running of the program can load the result instead of
training the recognizer again.
