"""
Detect eyes in an image.
"""

import cv2
import numpy as np
import BasicFunctions as bf

def findEyes(img, mode="haar", train=None, mask=None):
    """
    Two modes:
    1. haar -- uses the OpenCV built-in cascade classifier
    2. svm -- trains an SVM using a training/mask supplied by the user
    """

    if mode == "haar":
        return haarEyes(img)
    elif mode == "svm":
        return svmEyes(img, train, mask)
    else:
        raise Exception("Mode %s not supported." % mode)

def haarEyes(img):
    """ 
    Adapted from http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
    """

    haarpath = "/Users/davidfridovichkeil/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/haarcascade_eye.xml"

    img = (255.0 * img).astype(np.uint8)
    eye_cascade = cv2.CascadeClassifier(haarpath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    eye_centers = []
    eyes = eye_cascade.detectMultiScale(gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_centers.append((ex + ew/2.0, ey + eh/2.0))

    bf.imshow(img)
    return eye_centers

def svmEyes(img, train, mask):
    return []
