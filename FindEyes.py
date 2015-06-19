"""
Detect eyes in an image.
"""

import cv2
import numpy as np
import BasicFunctions as bf
from skimage import feature
from skimage import transform
from sklearn import svm as SVM
from sklearn import preprocessing
from collections import deque
from Queue import PriorityQueue
import sys

def findEyes(img, mode="haar", train=None, mask=None):
    """
    Two modes:
    1. haar -- uses the OpenCV built-in cascade classifier
    2. svm -- trains an SVM using a training image and eye patches
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

def svmEyes(img, training, eye_centers, eye_shape, 
            svm=None, scaler=None, locs=None):
    """ 
    SVM-based approach to detecting eyes. Input is as follows:
    * img -- new image to be searched for eyes
    * training -- old image used for generating an svm model
    * eyes_centers -- list of eye_centers used for generating an svm
    * eye_shape -- shape of eye patch
    * svm -- sklearn svm model; may be provided if it exists
    * scaler -- sklearn preprocessing scaler
    * locs -- approximate locations of eyes; used to speed up search process
    """

    img_gray = bf.rgb2gray(img)

    if svm is None:
        training_gray = bf.rgb2gray(training)
        eyes = []

        for ctr in eye_centers:
            eye_gray = extractTL(training_gray, 
                                 center2tl(ctr, eye_shape), eye_shape)
            eyes.append(eye_gray)

        # negative exemplars from rest of image
        negs = []
        num_negs = 0
        while num_negs < 100:
            tl = (np.random.randint(0, img.shape[0]), 
                  np.random.randint(0, img.shape[1]))
            if isValid(img, tl, eye_shape, eye_centers):
                num_negs += 1
                negs.append(extractTL(training_gray, tl, eye_shape))

        # create more positive exemplars by applying random small 3D rotations
        num_eyes = len(eyes)
        patches = deque([eye2patch(training_gray, eye1_tl, eye_shape),
                         eye2patch(training_gray, eye2_tl, eye_shape)])
        while num_eyes < 100:
            patch = patches.popleft()
            jittered = jitter(patch, eye_shape)
            patches.append(patch)
            eyes.append(patch2eye(jittered, eye_shape))
            num_eyes += 1

        # compute HOG for eyes and negs
        eyes_hog = []
        for eye in eyes:
            eyes_hog.append(feature.hog(eye))

        negs_hog = []
        for neg in negs:
            negs_hog.append(feature.hog(neg))

        # set up training dataset
        training_set = np.vstack((eyes_hog, negs_hog))

        training_labels = np.ones(num_eyes + num_negs)
        training_labels[num_eyes:] = 0
        
        scaler = preprocessing.StandardScaler().fit(training_set)
        training_set = scaler.transform(training_set)

        # train SVM
        weights = {0 : 1.0, 1 : 1.0}
        svm = SVM.SVC(C=1.0, gamma=0.1, kernel="rbf", class_weight=weights)
        svm.fit(training_set, training_labels)

    # find best matches, given svm and img_gray
    if locs

# Helper functions for svmEyes()
def center2tl(ctr, shape):
    return (ctr[0] - round(0.5*shape[0]), ctr[1] - round(0.5*shape[1]))

def tl2center(tl, shape):
    return (tl[0] + round(0.5*shape[0]), tl[1] - round(0.5*shape[1]))

def overlapsEye(tl, eye_centers, eye_shape):
    for ctr in eye_centers:
        eye_tl = center2tl(ctr, eye_shape)
        if not (((tl[0] < eye_tl[0]-eye_shape[0]) or 
                 (tl[0] > eye_tl[0]+eye_shape[0])) and
                ((tl[1] < eye_tl[1]-eye_shape[1]) or 
                 (tl[1] > eye_tl[1]+eye_shape[1]))
                return True
    return False

def isValid(img, tl, eye_shape, eye_centers):
    if (tl[0] < img.shape[0]-eye_shape[0] and 
        tl[1] < img.shape[1]-eye_shape[1] and not 
        overlapsEye(tl, eye_centers, eye_shape)):
        return True
    return False
    
def extractTL(img, tl, eye_shape):
    return img[tl[0]:tl[0]+eye_shape[0], tl[1]:tl[1]+eye_shape[1]]

def jitter(patch, eye_shape):
    f = 10.0
    
    # translate so center is at top left (origin)
    tf = np.matrix([[1, 0, -round(3.5 * eye_shape[1])],
                    [0, 1, -round(3.5 * eye_shape[0])],
                    [0, 0, 1]])
    
    # scale
    tf = np.matrix([[1/f, 0, 0],
                    [0, 1/f, 0],
                    [0, 0, 1]]) * tf
    
    # rotate about z
    theta = 0.05*np.random.randn()
    tf = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]]) * tf

    # now rotate about x
    theta = 0.05*np.random.randn()
    tf = np.matrix([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]]) * tf

    # now rotate about y
    theta = 0.05*np.random.randn()
    tf = np.matrix([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]]) * tf

    # scale back
    tf = np.matrix([[f, 0, 0],
                    [0, f, 0],
                    [0, 0, 1]]) * tf
    
    # translate back
    tf = np.matrix([[1, 0, round(3.5 * eye_shape[1])],
                    [0, 1, round(3.5 * eye_shape[0])],
                    [0, 0, 1]]) * tf
                    
    return transform.warp(patch, tf)

def eye2patch(img, tl, eye_shape):
    return img[tl[0]-3*eye_shape[0]:tl[0]+4*eye_shape[0], 
               tl[1]-3*eye_shape[1]:tl[1]+4*eye_shape[1]]

def patch2eye(patch, eye_shape):
    return patch[3*eye_shape[0]:-3*eye_shape[0], 
                 3*eye_shape[1]:-3*eye_shape[1]]

def testWindow(img, tl, svm, scaler):
    window = extractTL(img, tl, eye_shape)
    window_hog = scaler.transform(feature.hog(window))
    label = svm.predict(window_hog)
    score = svm.decision_function(window_hog)
    return score

def flat2ind(img, n, eye_shape):
    ncol = img.shape[1] - eye_shape[1]
    i = n / ncol
    j = n % ncol
    return (i, j)
