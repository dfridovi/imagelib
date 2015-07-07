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
	eye_locations = {"raw" : [], "filtered" : []}
	
	# initialize filters for eye locations
	l_filter = LocationPredictor(start=(350, 500))
	r_filter = LocationPredictor(start=(360, 675))

	try:

		while True:
			img, raw = fetchFrame(cam)
			locs = [l_filter.predict(), r_filter.predict()]
			found, scores = findEyes(img, svm, scaler, eye_shape, locs)

			print found

			# update filters
			if found[0][1] <= found[1][1]:
				l_filter.update(locs[0], scores[0])
				r_filter.update(locs[1], scores[1])
			else:
				l_filter.update(locs[1], scores[1])
				r_filter.update(locs[0], scores[0])

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

class LocationPredictor:
	""" Kalman filter to predict next eye locations."""

	def __init__(self, start, AX_SD=5.0, AY_SD=5.0):
		self.ax_var = AX_SD**2
		self.ay_var = AY_SD**2

		self.last_t = None
		self.xstate = np.matrix([[start[1]],
								 [0]], dtype=np.float)
		self.ystate = np.matrix([[start[0]],
								 [0]], dtype=np.float)

		# Kalman filter parameters
		self.F = None
		self.Qx = None
		self.Qy = None
		self.Px = np.matrix([[10**2, 0],
							 [0, 0]], dtype=np.float)
		self.Py = np.matrix([[10**2, 0],
							 [0, 0]], dtype=np.float)
		self.H = np.matrix([[1, 0]], dtype=np.float)


	def setF(self, dt):
		self.F = np.matrix([[1, dt],
						    [0, 1]], dtype=np.float)

	def setQ(self, dt):
		self.Qx = self.ax_var * np.matrix([[dt**4 * 0.25, dt**3 * 0.5],
						  	 			   [dt**3 * 0.5, dt**2]], dtype=np.float)
		self.Qy = self.ay_var * np.matrix([[dt**4 * 0.25, dt**3 * 0.5],
						  	 			   [dt**3 * 0.5, dt**2]], dtype=np.float)

	def predict(self):
		if self.last_t is not None:
			dt = time.time() - self.last_t
		else:
			dt = 0
		self.last_t = time.time()

		self.setF(dt)
		self.setQ(dt)

		self.xstate = self.F * self.xstate
		self.ystate = self.F * self.ystate
		self.Px = self.F * self.Px * self.F.T + self.Qx
		self.Py = self.F * self.Py * self.F.T + self.Qy

		return self.getPos()

	def update(self, loc, score):
		R = np.matrix([[np.abs(1.0 / score) * 100.0]], dtype=np.float)

		Yx = np.matrix([[loc[1]]], dtype=np.float) - self.H * self.xstate
		Yy = np.matrix([[loc[0]]], dtype=np.float) - self.H * self.ystate

		Sx = self.H * self.Px * self.H.T + R
		Sy = self.H * self.Py * self.H.T + R
		Kx = self.Px * self.H.T * np.linalg.inv(Sx)
		Ky = self.Py * self.H.T * np.linalg.inv(Sy)

		self.xstate += Kx * Yx
		self.ystate += Ky * Yy

		self.Px -= Kx * self.H * self.Px
		self.Py -= Ky * self.H * self.Py

	def getPos(self):
		return (self.ystate[0, 0], self.xstate[0, 0])


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