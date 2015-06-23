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

def findEyes(img, mode="haar", train=None, eye_centers=None, eye_shape=None,
             svm=None, scaler=None, locs=[]):
    """
    Two modes:
    1. haar -- uses the OpenCV built-in cascade classifier
    2. svm -- trains an SVM using a training image and eye patches
    """

    if mode == "haar":
        return haarEyes(img)
    elif mode == "svm":
        return svmEyes(img, train, eye_centers, eye_shape, svm, scaler, locs)
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
            svm=None, scaler=None, locs=[]):
    """ 
    SVM-based approach to detecting eyes. Input is as follows:
    * img -- new image to be searched for eyes
    * training -- old image used for generating an svm model
    * eyes_centers -- list of eye_centers used for generating an svm
    * eye_shape -- shape of eye patch
    * svm -- sklearn svm model; may be provided if it exists
    * scaler -- sklearn preprocessing scaler
    * locs -- approximate centers of eyes; used to speed up search process
    """

    img_gray = bf.rgb2gray(img)

    if svm is None:
        print "Building SVM classifier..."
        training_gray = bf.rgb2gray(training)
        eyes = []

        for ctr in eye_centers:
            eye_gray = extractTL(training_gray, 
                                 center2tl(ctr, eye_shape), eye_shape)
            eyes.append(eye_gray)

        # negative exemplars from rest of image
        print "Constructing negative exemplars..."
        negs = []
        num_negs = 0
        while num_negs < 1000:
            tl = (np.random.randint(0, img.shape[0]), 
                  np.random.randint(0, img.shape[1]))
            if (isValid(img, tl, eye_shape) and not
                overlapsEye(tl, eye_centers, eye_shape)):
                num_negs += 1
                negs.append(extractTL(training_gray, tl, eye_shape))

        # create more positive exemplars by applying random small 3D rotations
        print "Constructing positive exemplars..."
        num_eyes = len(eyes)
        patches = deque([eye2patch(training_gray, 
                                   center2tl(ctr, eye_shape), 
                                   eye_shape) for ctr in eye_centers])
                        
        while num_eyes < 1000:
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

        # set up training dataset (eyes = -1, negs = +1)
        training_set = np.vstack((negs_hog, eyes_hog))

        training_labels = np.ones(num_eyes + num_negs)
        training_labels[num_negs:] = -1
        
        scaler = preprocessing.StandardScaler().fit(training_set)
        training_set = scaler.transform(training_set)

        # train SVM
        print "Training SVM..."
        weights = {-1 : 1.0, 1 : 1.0}
        svm = SVM.SVC(C=5.0, kernel="linear", class_weight=weights)
        svm.fit(training_set, training_labels)

    # find best matches, given svm and img_gray
    detected = searchForEyes(img_gray, svm, scaler, eye_shape, locs)
    centers = []
    for tl in detected:
        centers.append(tl2center(tl, eye_shape))

    return centers

# Helper functions for svmEyes()
def searchForEyes(img, svm, scaler, eye_shape, locs=[]):
    """ Explore image starting at locs, visiting as few pixels as possible. """
    
    pq = PriorityQueue()
    tracker = MatchTracker()
    visited = np.zeros((img.shape[0]-eye_shape[0],
                        img.shape[1]-eye_shape[1]), dtype=np.bool)
    scores = np.zeros(visited.shape, dtype=np.float)

    # insert provided locations and 100 random locations around each one
    print "Seeding initial locations..."
    for loc in locs:
        tl = center2tl(loc, eye_shape)
        visited[loc[0], loc[1]] = True
        score = testWindow(img, svm, scaler, eye_shape, tl)[0]
        scores[loc[0], loc[1]] = score
        pq.put_nowait((score, tl))


        num_random = 0
        while (num_random < 50):
            tl = (loc[0] + np.random.randint(-25, 25), 
                  loc[1] + np.random.randint(-25, 25))
            if isValid(img, tl, eye_shape) and not visited[tl[0], tl[1]]:
                num_random += 1
                visited[tl[0], tl[1]] = True
                score = testWindow(img, svm, scaler, eye_shape, tl)[0]
                scores[tl[0], tl[1]] = score
                pq.put_nowait((score, tl))

        num_random = 0
        while (num_random < 100):
            tl = (loc[0] + np.random.randint(-50, 50), 
                  loc[1] + np.random.randint(-50, 50))
            if isValid(img, tl, eye_shape) and not visited[tl[0], tl[1]]:
                num_random += 1
                visited[tl[0], tl[1]] = True
                score = testWindow(img, svm, scaler, eye_shape, tl)[0]
                scores[tl[0], tl[1]] = score
                pq.put_nowait((score, tl))

                
    # insert 10 random locations
    print "Inserting 10 random locations..."
    num_random = 0
    while (num_random < 10):
        tl = (np.random.randint(0, img.shape[0]-eye_shape[0]),
              np.random.randint(0, img.shape[1]-eye_shape[1]))
        if not visited[tl[0], tl[1]]:
            num_random += 1
            visited[tl[0], tl[1]] = True
            score = testWindow(img, svm, scaler, eye_shape, tl)[0]
            scores[tl[0], tl[1]] = score
            pq.put_nowait((score, tl))

    # pick out the location with the best score
    best_score, best_tl = pq.get_nowait()
    
    # add 50 more random locations until best score is a match
    while best_score >= 0:
        print "Adding 50 more random locations..."
        num_random = 0
        while (num_random < 50):
            tl = (np.random.randint(0, img.shape[0]-eye_shape[0]),
                  np.random.randint(0, img.shape[1]-eye_shape[1]))
            if not visited[tl[0], tl[1]]:
                num_random += 1
                visited[tl[0], tl[1]] = True
                score = testWindow(img, svm, scaler, eye_shape, tl)[0]
                scores[tl[0], tl[1]] = score
                pq.put_nowait((score, tl))
        
        best_score, best_tl = pq.get_nowait()

    # stop when there are two good matches (score < -0.5) far enough apart
    # to be from two eyes, or after a maximum loop count
    loop_cnt = 0
    tracker.insert(best_score, best_tl)
    while (loop_cnt < 1000 and len(tracker.getBigClusters()) < 2):

        # look at unvisited pixels adjacent to current best_tl
        for tl in [(best_tl[0]-1, best_tl[1]), 
                   (best_tl[0]+1, best_tl[1]), 
                   (best_tl[0], best_tl[1]-1), 
                   (best_tl[0], best_tl[1]+1)]:
            if isValid(img, tl, eye_shape) and not visited[tl[0], tl[1]]:
                visited[tl[0], tl[1]] = True
                score = testWindow(img, svm, scaler, eye_shape, tl)[0]
                scores[tl[0], tl[1]] = score
                pq.put_nowait((score, tl))

        # get new best and update
        best_score, best_tl = pq.get_nowait()
        if best_score < 0:
            tracker.insert(best_score, best_tl)

        print "Current loop count: " + str(loop_cnt)
        print "Current cluster count: " + str(len(tracker.getBigClusters()))
        loop_cnt += 1

    if loop_cnt >= 1000:
        print "Did not find two good matches. Halting."

    tracker.printClusterScores()
    bf.imshow(visited, cbar=True)
    bf.imshow(scores, cbar=True)
    return tracker.getBigClusters()
    
class MatchTracker:
    """ Keep track of SVM matches, and do rudimentary clustering. """

    def __init__(self, MAX_DIST=50, MIN_AVGMASS=-0.1):
        self.clusters = {}
        self.MAX_DIST = MAX_DIST
        self.MIN_AVGMASS = MIN_AVGMASS

    def insert(self, score, location):
        best_centroid = None
        best_dist = float("inf")

        # search existing clusters for best match
        for centroid, stats in self.clusters.iteritems():
            d = dist(centroid, location)
            if d < best_dist:
                best_centroid = centroid
                best_dist = d

        if best_dist < self.MAX_DIST:
            old_size = self.clusters[best_centroid]["size"]
            old_mass = self.clusters[best_centroid]["total_mass"]
            new_mass = old_mass + score
            centroid = ((best_centroid[0]*old_mass + location[0]*score) / new_mass, 
                        (best_centroid[1]*old_mass + location[1]*score) / new_mass)
            del self.clusters[best_centroid]
            self.clusters[centroid] = {"total_mass" : new_mass, "size" : old_size + 1}

        # start new cluster
        else:
            self.clusters[location] = {"total_mass" : score, "size" : 1.0}

    def getBigClusters(self):
        big_clusters = []
        for centroid, stats in self.clusters.iteritems():
            if stats["total_mass"] / stats["size"] < self.MIN_AVGMASS:
                big_clusters.append(centroid)

        return big_clusters

    def printClusterScores(self):
        big_clusters = self.getBigClusters()
        for cluster in big_clusters:
            avg_mass = self.clusters[cluster]["total_mass"] / self.clusters[cluster]["size"]
            print str(cluster) + " : " + str(avg_mass)

def center2tl(ctr, shape):
    return (ctr[0] - round(0.5*shape[0]), ctr[1] - round(0.5*shape[1]))

def tl2center(tl, shape):
    return (tl[0] + round(0.5*shape[0]), tl[1] + round(0.5*shape[1]))

def dist(coords1, coords2):
    return np.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2)

def overlapsEye(tl, eye_centers, eye_shape):
    for ctr in eye_centers:
        eye_tl = center2tl(ctr, eye_shape)
        if not (((tl[0] < eye_tl[0]-eye_shape[0]) or 
                 (tl[0] > eye_tl[0]+eye_shape[0])) and
                ((tl[1] < eye_tl[1]-eye_shape[1]) or 
                 (tl[1] > eye_tl[1]+eye_shape[1]))):
                return True
    return False

def isValid(img, tl, eye_shape):
    if (tl[0] < img.shape[0]-eye_shape[0] and 
        tl[1] < img.shape[1]-eye_shape[1] and
        tl[0] >= 0 and tl[1] >= 0):
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
    theta = 0.2*np.random.randn()
    tf = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]]) * tf

    # now rotate about x
    theta = 0.1*np.random.randn()
    tf = np.matrix([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]]) * tf

    # now rotate about y
    theta = 0.1*np.random.randn()
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

def testWindow(img, svm, scaler, eye_shape, tl):
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
