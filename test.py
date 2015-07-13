"""
Test file to demonstrate imagelib functionality.
"""

import numpy as np
import BasicFunctions as bf
from Sharpening import sharpen
from Blurring import blur
from FindEyes import searchForEyesSVM, createSVM
import time, os
import cPickle as pickle
from TrackEyes import trackEyes

# import image
#img = bf.imread("lotr.JPG")
#img = bf.imread("eye.png")
#img = bf.imread("obama.jpg")
img = bf.imread("me.jpg")

# test blurring
#blurred = blur(img, mode="gaussian", k=5)
#bf.imshow(blurred)

# test sharpening
#sharpened = sharpen(img, k=21, lo_pass=True, min_diff=0.01, alpha=3.0)
#bf.imshow(sharpened)

# test exposure adjustment
# darker = bf.adjustExposure(img, gamma=1.5)
# bf.imshow(darker)
# lighter = bf.adjustExposure(img, gamma=0.5)
# bf.imshow(lighter)

# test eye detection
# eye_shape = (25, 50)
# eye1_tl = (200, 480)
# eye2_tl = (195, 655)
# eye1_ctr = bf.tl2center(eye1_tl, eye_shape)
# eye2_ctr = bf.tl2center(eye2_tl, eye_shape)
# svm, scaler = createSVM(training=img, eye_centers=[eye1_ctr, eye2_ctr], eye_shape=eye_shape) 
# eyes = searchForEyesSVM(img=img, svm=svm, scaler=scaler, eye_shape=eye_shape, locs=[(190, 470), (190, 650)])
# print eyes

# test video eye detection
eye1_tl = (344, 409)
eye2_tl = (348, 515)
eye_shape = (24, 48)
eye1_ctr = bf.tl2center(eye1_tl, eye_shape)
eye2_ctr = bf.tl2center(eye2_tl, eye_shape)
eyes = [eye1_ctr, eye2_ctr]

if os.path.isfile("svm.pkl"):
    svm_file = open("svm.pkl", "rb")
    (svm, scaler) = pickle.load(svm_file)
    svm_file.close()
else: 
	svm, scaler = createSVM(training=img, eye_centers=eyes, eye_shape=eye_shape) 
	svm_file = open("svm.pkl", "wb")
	pickle.dump((svm, scaler), svm_file)
	svm_file.close()

trackEyes(svm=svm, scaler=scaler)