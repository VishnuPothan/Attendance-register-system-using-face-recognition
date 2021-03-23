# import the necessary packages
from __future__ import print_function
from photoboothapp import PhotoBoothApp
from imutils.video import VideoStream
import argparse
import time
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream().start()
time.sleep(2.0)

detectorPath = "model"
embeddingModelPath = "model/openface_nn4.small2.v1.t7"
recognizerPath = "output/face_recognizer.pickle"
labelPath = "output/face_label.pickle"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModelPath)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizerPath, "rb").read())
le = pickle.loads(open(labelPath, "rb").read())

# start the app
pba = PhotoBoothApp(vs, args["output"], detector, embedder, recognizer, le)
pba.root.mainloop()