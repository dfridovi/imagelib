"""
Detect eyes in an image.
"""

import cv2
import numpy as np
import BasicFunctions as bf
from skimage import transform
from sklearn import svm as SVM
from sklearn import preprocessing
from collections import deque
from Queue import PriorityQueue
import sys

def haarEyes(gray, haar_classifier=None):
    """ 
    Adapted from OpenCV online tutorials.
    """

    if haar_classifier is not None:
        haarpath = ("/Users/davidfridovichkeil/anaconda/pkgs/" +
                    "opencv-2.4.8-np17py27_2/" +
                    "share/OpenCV/haarcascades/haarcascade_eye.xml")
        haar_classifier = cv2.CascadeClassifier(haarpath)
    
    gray = (255.0 * gray).astype(np.uint8)

    eye_centers = []
    eyes = haar_classifier.detectMultiScale(gray)

    for (ex, ey, ew, eh) in eyes:
        eye_centers.append((ey + eh/2.0, ex + ew/2.0))

    return eye_centers

def createSVM(training, eye_centers, eye_shape):
    """ 
    Create SVM model for eye detection. Inputs are as follows:
    * training -- old image used for generating an svm model
    * eyes_centers -- list of eye_centers used for generating an svm
    * eye_shape -- shape of eye patch
    """

    print "Building SVM classifier..."
    training_gray = bf.rgb2gray(training)
    eyes = []

    for ctr in eye_centers:
        eye_gray = extractTL(training_gray, 
                             bf.center2tl(ctr, eye_shape), eye_shape)
        eyes.append(eye_gray)

    # negative exemplars from rest of image
    print "Constructing negative exemplars..."
    negs = []
    num_negs = 0
    while num_negs < 999:
        tl = (np.random.randint(0, training_gray.shape[0]), 
              np.random.randint(0, training_gray.shape[1]))
        if (isValid(training_gray, tl, eye_shape) and not
            overlapsEye(tl, eye_centers, eye_shape)):
            num_negs += 1
            negs.append(extractTL(training_gray, tl, eye_shape))

    # create more positive exemplars by applying random small 3D rotations
    print "Constructing positive exemplars..."
    num_eyes = len(eyes)
    patches = deque([eye2patch(training_gray, 
                               bf.center2tl(ctr, eye_shape), 
                               eye_shape) for ctr in eye_centers])
                    
    while num_eyes < 999:
        patch = patches.popleft()
        jittered = jitter(patch, eye_shape)
        patches.append(patch)
        new_eye = patch2eye(jittered, eye_shape)
        eyes.append(new_eye)
        num_eyes += 1

        # change lighting conditions
        eyes.append(bf.adjustExposure(new_eye, 0.75))
        eyes.append(bf.adjustExposure(new_eye, 0.9))
        num_eyes += 2

    # compute HOG for eyes and negs
    eyes_hog = []
    for eye in eyes:
        eyes_hog.append(bf.getHog(eye))

    negs_hog = []
    for neg in negs:
        negs_hog.append(bf.getHog(neg))

    # set up training dataset (eyes = -1, negs = +1)
    training_set = np.vstack((negs_hog, eyes_hog))

    training_labels = np.ones(num_eyes + num_negs)
    training_labels[num_negs:] = -1
    
    scaler = preprocessing.StandardScaler().fit(training_set)
    training_set = scaler.transform(training_set)

    # train SVM
    print "Training SVM..."
    weights = {-1 : 1.0, 1 : 1.0}
    svm = SVM.SVC(C=1.0, gamma=0.01, kernel="rbf", class_weight=weights)
    svm.fit(training_set, training_labels)

    return svm, scaler

def searchForEyesSVM(gray, svm, scaler, eye_shape, locs=[]):
    """ 
    Explore image on the cell level, reducing HOG calculations.
    Inputs are as follows (besides the obvious)
    * svm -- sklearn svm model; may be provided if it exists
    * scaler -- sklearn preprocessing scaler
    * locs -- list of approximate centers of eyes
    * eye_shape -- size of eye template in pixels (rows, columns)
    """

    tracker = MatchTracker()
    pq = PriorityQueue()

    eye_cells = (eye_shape[0] // 8, eye_shape[1] // 8)
    hog_computed = np.zeros((gray.shape[0] // 8, gray.shape[1] // 8),
                            dtype=np.bool)

    # distribution parameters
    blind_skip = 3

    # adjust locs
    locs[0] = (int(locs[0][0]), int(locs[0][1]))
    locs[1] = (int(locs[1][0]), int(locs[1][1]))

    # only compute HOG on subset of image at first
    min_x = min(bf.center2tl(locs[0], eye_shape)[1], 
                bf.center2tl(locs[1], eye_shape)[1])
    max_x = max(bf.center2tl(locs[0], eye_shape)[1], 
                bf.center2tl(locs[1], eye_shape)[1])
    min_y = min(bf.center2tl(locs[0], eye_shape)[0], 
                bf.center2tl(locs[1], eye_shape)[0])
    max_y = max(bf.center2tl(locs[0], eye_shape)[0], 
                bf.center2tl(locs[1], eye_shape)[0])
    
    tl = (min_y - 4*eye_shape[0], min_x - 4*eye_shape[1])
    br = (max_y + 4*eye_shape[0], max_x + 4*eye_shape[1])

    tl_cell = bf.px2cell(tl)
    br_cell = bf.px2cell(br)

    tl = bf.cell2px(tl_cell)
    br = bf.cell2px(br_cell)

    indices = np.index_exp[tl_cell[0]:br_cell[0], tl_cell[1]:br_cell[1], :]
    indices_computed = np.index_exp[tl_cell[0]:br_cell[0], tl_cell[1]:br_cell[1]]

    hog = np.empty((gray.shape[0] // 8, gray.shape[1] // 8, 9), 
                   dtype=np.float)
    hog[indices] = bf.getHog(gray[tl[0]:br[0], tl[1]:br[1]], 
                             normalize=False, flatten=False)
    hog_computed[indices_computed] = True

    # create visited array
    visited = np.zeros((hog.shape[0]-eye_cells[0]+1,
                        hog.shape[1]-eye_cells[1]+1), dtype=np.bool)
 
    # insert provided locations and begin exploration around each one
    for loc in locs:
        tl = bf.center2tl(loc, eye_shape)
        tl = bf.px2cell(tl)

        # only proceed if valid
        if not isValid(hog, tl, eye_cells):
            continue

        # handle this point
        visited[tl[0], tl[1]] = True
        score = testWindow(hog, svm, scaler, eye_cells, tl)[0]
        pq.put_nowait((score, tl))

        if score <= 0:
            tracker.insert(score, tl)

    # search
    greedySearch(hog, hog_computed, svm, scaler, 
                 eye_cells, visited, tracker, pq)
    if tracker.isDone():
        tracker.printClusterScores()
        clusters, scores = tracker.getBigClusters()
        centers = cellTLs2ctrs(clusters, eye_shape)
        return centers, scores

    # if needed, repeat above search technique, but with broader scope
    print "Searching blindly."

    hog = bf.getHog(gray, normalize=False, flatten=False)
    hog_computed[:, :] = True

    for i in range(20, visited.shape[0]-20, blind_skip):
        for j in range(20, visited.shape[1]-20, blind_skip):
            test = (i, j)

            # only proceed if valid and not visited
            if (not isValid(hog, test, eye_cells)) or visited[i, j]:
                continue

            # handle this point
            visited[i, j] = True
            score = testWindow(hog, svm, scaler, eye_cells, test)[0]
            pq.put_nowait((score, test))

            if score <= 0:
                tracker.insert(score, test)

    greedySearch(hog, hog_computed, svm, scaler, 
                 eye_cells, visited, tracker, pq) 
    if tracker.isDone():
        tracker.printClusterScores()
        clusters, scores = tracker.getBigClusters()
        centers = cellTLs2ctrs(clusters, eye_shape)
        return centers, scores

    print "Did not find two good matches."
    tracker.printClusterScores()
    clusters, scores = tracker.getTwoBestClusters()
    centers = cellTLs2ctrs(clusters, eye_shape)
    return centers, scores


def greedySearch(hog, hog_computed, svm, scaler, eye_cells, 
                 visited, tracker, pq):
    """ Greedy search algorithm. """

    cnt = 0              
    max_iter = int(hog_computed.sum() * 0.9)

    while (not tracker.isDone()) and (cnt < max_iter) and (pq.qsize() > 0):
        best_score, best_tl = pq.get_nowait()

        for test in [(best_tl[0]-1, best_tl[1]), 
                     (best_tl[0]+1, best_tl[1]), 
                     (best_tl[0], best_tl[1]-1), 
                     (best_tl[0], best_tl[1]+1)]:
            if (isValid(hog, test, eye_cells) and 
                hog_computed[test[0], test[1]] and
                not visited[test[0], test[1]]):

                visited[test[0], test[1]] = True
                score = testWindow(hog, svm, scaler, eye_cells, test)[0]
                
                pq.put_nowait((score, test))
                cnt += 1

                if score <= 0:
                    tracker.insert(score, test)

        # release pq
        pq.task_done()

class MatchTracker:
    """ Keep track of SVM matches, and do rudimentary clustering. """

    def __init__(self, MAX_DIST=3, MAX_MASS=-2.0, MIN_SIZE=4):
        self.clusters = {}
        self.MAX_DIST = MAX_DIST
        self.MAX_MASS = MAX_MASS
        self.MIN_SIZE = MIN_SIZE

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
            self.clusters[centroid] = {"total_mass" : new_mass, 
                                       "size" : old_size + 1}

        # start new cluster
        else:
            self.clusters[location] = {"total_mass" : score, "size" : 1.0}

    def getBigClusters(self):
        big_clusters = []
        scores = []

        for centroid, stats in self.clusters.iteritems():
            if (stats["total_mass"] < self.MAX_MASS and 
                stats["size"] > self.MIN_SIZE):
                big_clusters.append(centroid)
                scores.append(stats["total_mass"])

        return big_clusters, scores

    def getTwoBestClusters(self):
        best_clusters = PriorityQueue()
        cnt = 0

        for centroid, stats in self.clusters.iteritems():
            avg_mass = stats["total_mass"] / stats["size"]
            best_clusters.put_nowait((avg_mass, centroid))
            cnt += 1

        avg_mass, cluster1 = best_clusters.get_nowait(); best_clusters.task_done()
        avg_mass, cluster2 = best_clusters.get_nowait(); best_clusters.task_done()

        centroids = [cluster1, cluster2]
        scores = [self.clusters[cluster1]["total_mass"],
                  self.clusters[cluster2]["total_mass"]]

        print "Total clusters: " + str(cnt)
        return centroids, scores

    def printClusterScores(self):
        big_clusters, scores = self.getBigClusters()

        for cluster in big_clusters:
            mass = self.clusters[cluster]["total_mass"]
            size = self.clusters[cluster]["size"]
            print str(cluster) + " : " + str((mass, size))

    def isDone(self):
        big_clusters, scores = self.getBigClusters()
        if len(big_clusters) < 2:
            return False
        return True

def dist(coords1, coords2):
    return np.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2)

def cellTLs2ctrs(cellTLs, eye_shape):
    ctrs = []
    for cellTL in cellTLs:
        ctrs.append(bf.tl2center(bf.cell2px(cellTL), eye_shape))

    return ctrs

def overlapsEye(tl, eye_centers, eye_shape):
    for ctr in eye_centers:
        eye_tl = bf.center2tl(ctr, eye_shape)
        if not (((tl[0] < eye_tl[0]-eye_shape[0]) or 
                 (tl[0] > eye_tl[0]+eye_shape[0])) and
                ((tl[1] < eye_tl[1]-eye_shape[1]) or 
                 (tl[1] > eye_tl[1]+eye_shape[1]))):
                return True
    return False

def isValid(img, tl, shape):
    if (tl[0] <= img.shape[0]-shape[0] and 
        tl[1] <= img.shape[1]-shape[1] and
        tl[0] >= 0 and tl[1] >= 0):
        return True
    return False
    
def extractTL(img, tl, eye_shape):
    return img[tl[0]:tl[0]+eye_shape[0], tl[1]:tl[1]+eye_shape[1]]

def jitter(patch, eye_shape):
    f = 100.0
    
    # translate so center is at top left (origin)
    tf = np.matrix([[1, 0, -round(4.5 * eye_shape[1])],
                    [0, 1, -round(4.5 * eye_shape[0])],
                    [0, 0, 1]])
    
    # scale
    tf = np.matrix([[1/f, 0, 0],
                    [0, 1/f, 0],
                    [0, 0, 1]]) * tf
    
    # rotate about z
    theta = 0.1*np.random.randn()
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
    tf = np.matrix([[1, 0, round(4.5 * eye_shape[1])],
                    [0, 1, round(4.5 * eye_shape[0])],
                    [0, 0, 1]]) * tf
                    
    return transform.warp(patch, tf)

def eye2patch(img, tl, eye_shape):
    return img[tl[0]-4*eye_shape[0]:tl[0]+5*eye_shape[0], 
               tl[1]-4*eye_shape[1]:tl[1]+5*eye_shape[1]]

def patch2eye(patch, eye_shape):
    return patch[4*eye_shape[0]:-4*eye_shape[0], 
                 4*eye_shape[1]:-4*eye_shape[1]]

def testWindow(hog, svm, scaler, eye_cells, tl):
    window = hog[tl[0]:tl[0]+eye_cells[0], tl[1]:tl[1]+eye_cells[1], :]
    window = scaler.transform(bf.normalizeHog(window).ravel())
    score = svm.decision_function(window)
    return score