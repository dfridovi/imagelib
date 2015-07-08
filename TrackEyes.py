"""
Eye tracking in video.
"""

import cv2
import numpy as np
import threading, os
from Queue import Queue
import BasicFunctions as bf
from FindEyes import haarEyes, createSVM, searchForEyesSVM
import sys, time
import cPickle as pickle
from LocationPredictors import LocationPredictorLPF

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
	eye_locations = {"raw" : [], "filtered" : []}
	
	# initialize filters for eye locations
	l_filter = LocationPredictorLPF(start=(350, 500))
	r_filter = LocationPredictorLPF(start=(360, 675))

	try:

		while True:
			img, raw = fetchFrame(cam)
			locs = [l_filter.predict(), r_filter.predict()]
			found, scores = findEyes(img, svm, scaler, eye_shape, locs)

			print found

			# update filters
			if found[0][1] <= found[1][1]:
				l_filter.update(found[0], scores[0])
				r_filter.update(found[1], scores[1])
			else:
				l_filter.update(found[1], scores[1])
				r_filter.update(found[0], scores[0])

			# get filtered locations and visualize raw/filtered
			filtered = [l_filter.getPos(), r_filter.getPos()]
			visualizeEyes(raw, found, filtered, eye_locations, eye_shape)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				raise(KeyboardInterrupt)

	# When everything done, release the capture and pickle the data
	except KeyboardInterrupt:
		print "\nKeyboardInterrupt: Cleaning up and exiting."
		
		cam.release()
		cv2.destroyAllWindows()

		out_file = open(out_file, "wb")
		pickle.dump(eye_locations, out_file)
		out_file.close()


def findEyes(img, svm, scaler, eye_shape, locs):
	""" Find eyes in the given frame. """

	# find eyes
	found, scores = searchForEyesSVM(img, svm, scaler, eye_shape, locs)
	return found, scores

def fetchFrame(cam):
	""" Fetch new frame from camera and preprocess. """

	ret, frame = cam.read()

	# handle invalid return value
	if not ret:
		print "Error. Invalid return value."
		sys.exit()

	# convert to grayscale
	img = bf.rescale(bf.bgr2gray(frame))

	return img, frame

def visualizeEyes(raw, found, filtered, eye_locations, eye_shape):
	""" Display found eyes, and save. """

	# save data
	eye_locations["raw"].append(found)
	eye_locations["filtered"].append(filtered)

	# display 
	bf.drawRectangle(raw, found[0], eye_shape, (0, 0, 255))
	bf.drawRectangle(raw, found[1], eye_shape, (0, 0, 255))
	bf.drawRectangle(raw, (int(filtered[0][0]), int(filtered[0][1])), 
					 eye_shape, (255, 0, 0))
	bf.drawRectangle(raw, (int(filtered[1][0]), int(filtered[1][1])), 
					 eye_shape, (255, 0, 0))

	cv2.imshow("camera", raw)