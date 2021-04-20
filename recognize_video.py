# USAGE

# python recognize_video.py --detector model --embedding-model model/openface_nn4.small2.v1.t7 --recognizer output/face_recognizer.pickle --le output/face_label.pickle

# import packages
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import date
from env_variables import firebaseConfig
import numpy as np
import argparse
import imutils
import pickle
import pyrebase
import time
from datetime import datetime
import cv2
import os
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.9,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initlaize firebase connection
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# DETAILS FETCHED FROM FIREBASE
teacherDetailsDict = db.child('teacher').get().val()
studentDetailsDict  = db.child('student').get().val()


# start the FPS throughput estimator
fps = FPS().start()

# teacher found
record_atendance = False
attendance_list = []
recorded = []
subject = ""

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=640)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Identify details       
            if(record_atendance):
                # draw the bounding box
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                for key, value in studentDetailsDict.items():
                    if(value["ID"] == name):
                        if(name not in recorded):
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            student_dict = {"ID" : name, "name" : value["name"], "time" : str(current_time), "subject" : subject}
                            attendance_list.append(student_dict)
                            recorded.append(name)
            else:
                # check if teacher
                for key, value in teacherDetailsDict.items():
                    if(value["ID"] == name):
                        # draw the bounding box
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                        print(value)
                        verify = input("Record attendance(yes/no) : ")

                        if(verify == "yes"):
                            record_atendance = True
                            subject = value["subject"]

    # update the FPS counter
    fps.update()
    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `s` key was pressed, stop attendance recording
    if key == ord("s"):
        print(attendance_list)
        attendance_dict = {subject : attendance_list}
        db.child("attendance").child(str(date.today())).push(attendance_dict)
        record_atendance = False
        attendance_list = []
        recorded = []
        subject = ""
        
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print(attendance_list)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup

vs.stream.release()
cv2.destroyAllWindows()