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

class LocationPredictor:
	""" Kalman filter to predict next eye locations."""

	def __init__(self, shape, AX_SD=100.0, AY_SD=100.0):
		self.ax_var = AX_SD**2
		self.ay_var = AY_SD**2

		self.last_t = None
		self.xstate = np.matrix([[0.5 * shape[1]],
									  [0]], dtype=np.float)
		self.ystate = np.matrix([[0.5 * shape[0]],
									  [0]], dtype=np.float)

		# Kalman filter parameters
		self.F = None
		self.Q = None
		self.P = np.matrix([[200**2, 0],
							[0, 200**2]], dtype=np.float)
		self.H = np.matrix([[1, 0]], dtype=np.float)
		self.K = None
		self.R = None
		self.S = None
		self.Y = None


	def setF(self, dt):
		self.F = np.matrix([[1, dt],
						    [0, 1]], dtype=np.float)

	def setQ(self, dt):
		self.Q = np.matrix([[dt**4 * 0.25, dt**3 * 0.5],
						  	[dt**3 * 0.5, dt**2]], dtype=np.float)

	def setR(self, score):
		self.R = np.matrix([[np.abs(1.0 / score) * 10.0]], dtype=np.float)

	def predict(self):
		if self.last_t is not None:
			dt = time.time() - self.last_t
		else:
			dt = 0

		self.setF(dt)
		self.setQ(dt)

		self.xstate = self.F * self.xstate
		self.ystate = self.F * self.ystate
		self.P = self.F * self.P * self.F.T + self.Q

	def update(self, loc, score):
		if self.last_t is not None:
			dt = time.time() - self.last_t
		else:
			dt = 0
		self.last_t = time.time()

		self.Y = 


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