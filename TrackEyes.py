"""
Eye tracking in video.
"""

import cv2
import numpy as np
import cPickle as pickle
import threading, os
from Queue import Queue
import BasicFunctions as bf
from FindEyes import haarEyes, createSVM, searchForEyesSVM
import sys

def trackEyes(video_source=0, eye_shape=(24, 48),
			  svm_file="svm.pkl", out_file="eye_tracking_data.pkl"):
	"""
	Track eyes in video, and output position data to file. 
	Inputs are as follows:
	* video_source -- which file to read video from (default is 0, or webcam)
	* eye_shape -- shape of eye patch (rows, columns)
	* svm_file -- can provide source file for svm (and scaler)
	* out_file -- where to save the data
	"""

	# read in svm, scaler
	if os.path.isfile(svm_file):
		svm_file = open(svm_file, "rb")
		(svm, scaler) = pickle.load(svm_file)
		svm_file.close()
	else:
		print "Error. Must provide valid svm_file."
		sys.exit(1)

	# initialize camera
	cam = cv2.VideoCapture(video_source)

	# store the data
	eye_locations = {"index" : [], "locs" : [], "next_frame" : 1}

	# initialize queues
	video_feed = Queue()
	raw_feed = Queue()
	found_eyes = Queue()
	
	# initialize container for previous eyes
	last_eyes = []

	# initialize threads
	frame_fetcher = StoppableThread(target=fetchFrame, 
									args=(cam, video_feed, raw_feed))
	eye_finder = StoppableThread(target=findEyes, 
								 args=(video_feed, found_eyes, svm, scaler, 
									   eye_shape, last_eyes, frame_fetcher))

	try:
		frame_fetcher.daemon = True
		eye_finder.daemon = True

		frame_fetcher.start()
		eye_finder.start()

		while True:
			visualizeEyes(raw_feed, found_eyes, eye_locations, eye_shape, eye_finder)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				raise(KeyboardInterrupt)

	# When everything done, release the capture and pickle the data
	except KeyboardInterrupt:
		print "\nKeyboardInterrupt: Cleaning up and exiting."

		frame_fetcher.stop()
		eye_finder.stop()
		
		out_file = open(out_file, "wb")
		pickle.dump(eye_locations, out_file)
		out_file.close()

		frame_fetcher.join()
		eye_finder.join()

		cv2.destroyAllWindows()
		cam.release()


class StoppableThread(threading.Thread):
    """
    Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.

    From: http://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
    """

    def __init__(self, target, args):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()
        self._target = target
        self._args = args

    def stop(self):
        self._stop.set()

    def go(self):
    	self._stop.clear()

    def run(self):
    	if not self._stop.isSet():
    		self._target(self._args)
    		self._stop.set()

    def stopped(self):
        return self._stop.isSet()


def findEyes((video_feed, found_eyes, svm, scaler, 
			  eye_shape, last_eyes, frame_fetcher)):
	""" Find eyes in the given frame, and put output on queue. """

	print "finding eyes"

	# fetch new image and last eye locations
	img = video_feed.get()

	if not frame_fetcher.isAlive():
		frame_fetcher.go()
		frame_fetcher.run()

	# find eyes and add to queue
	last_eyes = searchForEyesSVM(img, svm, scaler, eye_shape, last_eyes)
	found_eyes.put(last_eyes)

	# release video feed
	video_feed.task_done()

def fetchFrame((cam, video_feed, raw_feed)):
	""" Fetch new frame from camera, then process and put on queue. """

	ret, frame = cam.read()
	print "fetching new frame"
	# handle invalid return value
	if not ret:
		print "Error. Invalid return value."
		sys.exit()

	# convert to grayscale
	img = bf.rescale(bf.bgr2gray(frame))

	# append to queues
	video_feed.put(img)
	raw_feed.put(frame)

def visualizeEyes(raw_feed, found_eyes, eye_locations, eye_shape, eye_finder):
	""" Display found eyes, and save. """

	print "displaying"

	# fetch raw frame and eye locations
	frame = raw_feed.get()
	found = found_eyes.get()

	if not eye_finder.isAlive():
		eye_finder.go()
		eye_finder.run()

	# save data
	eye_locations["index"].append(eye_locations["next_frame"])
	eye_locations["locs"].append(found)
	eye_locations["next_frame"] += 1

	# display 
	for eye in found:
		bf.drawRectangle(frame, eye, eye_shape, (0, 255, 0))

	cv2.imshow("camera", frame)

	# release queues
	raw_feed.task_done()
	found_eyes.task_done()