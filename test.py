"""
Test file to demonstrate imagelib functionality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import BasicFunctions as bf
from Sharpening import sharpen
from Blurring import blur
from FindEyes import findEyes, searchForEyesSVM, createSVM
import time

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
eye1_tl = (348, 409)
eye2_tl = (353, 515)
eye_shape = (24, 46)
eye1_ctr = bf.tl2center(eye1_tl, eye_shape)
eye2_ctr = bf.tl2center(eye2_tl, eye_shape)
eyes = [eye1_ctr, eye2_ctr]

svm, scaler = createSVM(training=img, eye_centers=eyes, eye_shape=eye_shape) 

cap = cv2.VideoCapture(0)
start = time.time()
cnt = 0

try:
	while True:
		ret, frame = cap.read()
		img = np.zeros(frame.shape)
		img[:, :, 0] = frame[:, :, 2]
		img[:, :, 1] = frame[:, :, 1]
		img[:, :, 2] = frame[:, :, 0]

		found = searchForEyesSVM(img=img, svm=svm, scaler=scaler, 
								 eye_shape=eye_shape, locs=eyes)

		if len(found) == 2:
			eyes = found

			# draw rectangles
			for eye in eyes:
				bf.drawRectangle(frame, eye, eye_shape, (0, 1, 0))

		cv2.imshow("camera", frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		cnt += 1

# When everything done, release the capture
except KeyboardInterrupt:
	cap.release()
	cv2.destroyAllWindows()

	print "Average seconds per frame: " + str((time.time() - start) / cnt)