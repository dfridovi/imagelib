"""
Eye tracking in video.
"""

import cv2
import numpy as np
import threading, os
from Queue import Queue
import BasicFunctions as bf
from FindEyes import haarEyes, createSVM, searchForEyesSVM
import sys
import cPickle as pickle

def trackEyes(svm, scaler, video_source=0, eye_shape=(24, 48),
			  out_file="eye_tracking_data.pkl"):
	"""
	Track eyes in video, and output position data to file. 
	Inputs are as follows:
	* video_source -- which file to read video from (default is 0, or webcam)
	* eye_shape -- shape of eye patch (rows, columns)
	* svm, scaler -- svm and data preprocessor
	* out_file -- where to save the data
	"""

	# initialize camera
	cam = cv2.VideoCapture(video_source)

	# store the data
	eye_locations = {"index" : [], "locs" : [], "next_frame" : 1}
	
	# initialize container for previous eyes
	found = []

	try:

		while True:
			img, raw = fetchFrame(cam)
			found = findEyes(img, svm, scaler, eye_shape, found)
			visualizeEyes(raw, found, eye_locations, eye_shape)

			# if cv2.waitKey(1) & 0xFF == ord("q"):
			# 	raise(KeyboardInterrupt)

	# When everything done, release the capture and pickle the data
	except KeyboardInterrupt:
		print "\nKeyboardInterrupt: Cleaning up and exiting."
		
		cam.release()
		cv2.destroyAllWindows()

		out_file = open(out_file, "wb")
		pickle.dump(eye_locations, out_file)
		out_file.close()


def findEyes(img, svm, scaler, eye_shape, locs):
	""" Find eyes in the given frame, and put output on queue. """

	# find eyes and add to queue
	found = searchForEyesSVM(img, svm, scaler, eye_shape, locs)
	return found

def fetchFrame(cam):
	""" Fetch new frame from camera, then process and put on queue. """

	ret, frame = cam.read()

	# handle invalid return value
	if not ret:
		print "Error. Invalid return value."
		sys.exit()

	# convert to grayscale
	img = bf.rescale(bf.bgr2gray(frame))

	return img, frame

def visualizeEyes(raw, found, eye_locations, eye_shape):
	""" Display found eyes, and save. """

	# save data
	eye_locations["index"].append(eye_locations["next_frame"])
	eye_locations["locs"].append(found)
	eye_locations["next_frame"] += 1

	# display 
	for eye in found:
		bf.drawRectangle(raw, eye, eye_shape, (0, 255, 0))

	cv2.imshow("camera", raw)