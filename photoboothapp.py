# import the necessary packages
from __future__ import print_function
from PIL import Image as ImagePIL
from PIL import ImageTk
import tkinter as tki
from tkinter import *
import numpy as np
import threading
import datetime
import imutils
import json
import cv2
import os

class PhotoBoothApp:
	def __init__(self, vs, outputPath, detector, embedder, recognizer, le):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None
		self.detector = detector
		self.embedder = embedder
		self.recognizer = recognizer
		self.le = le
		self.i = 4

		# initialize the root window and image panel
		self.rootAttendance = tki.Toplevel()
		self.panel = None
		self.labelAttendance = None

		# create a button, that when pressed, will take the current
		# frame and save it to file
		self.btn = tki.Button(self.rootAttendance, text="Stop Recording", command =lambda : self.stopEvent.set()).grid(row=1, column=0)

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.rootAttendance.wm_title("Attendance Record")
		self.rootAttendance.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def DisplayAttendance(self, studentDetailsList):
		for j in range(len(studentDetailsList)): 
			self.e = Entry(self.rootAttendance, width=20, fg='powder blue', font=('Arial',12,'bold')) 
			self.e.grid(row=self.i+3, column=j) 
			self.e.insert(END, studentDetailsList[j])
		self.i += 1

	def videoLoop(self):
		try:
			# teacher found
			record_atendance = False
			attendance_list = []
			recorded = []
			subject = ""

			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():

				# grab the frame from the video stream and resize it to
				# have a maximum width of 640 pixels
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=300)
				frameLoc = self.frame
				(h, w) = frameLoc.shape[:2]

				# construct a blob from the image
				imageBlob = cv2.dnn.blobFromImage( cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

				# apply OpenCV's deep learning-based face detector to localize
				# faces in the input image
				self.detector.setInput(imageBlob)
				detections = self.detector.forward()
				
				# loop over the detections
				for i in range(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated with
					# the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections
					if confidence > 0.6:
						# compute the (x, y)-coordinates of the bounding box for
						# the face
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						# extract the face ROI
						face = self.frame[startY:endY, startX:endX]
						(fH, fW) = face.shape[:2]

						# ensure the face width and height are sufficiently large
						if fW < 20 or fH < 20:
							continue

						# construct a blob for the face ROI, then pass the blob
						# through our face embedding model to obtain the 128-d
						# quantification of the face
						faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
														(96, 96), (0, 0, 0), swapRB=True, crop=False)
						self.embedder.setInput(faceBlob)
						vec = self.embedder.forward()

						# perform classification to recognize the face
						preds = self.recognizer.predict_proba(vec)[0]
						j = np.argmax(preds)
						proba = preds[j]
						name = self.le.classes_[j]

						# teacher face check
						with open('dataset/teacher_data.json') as teacherFile:
							data = json.load(teacherFile)
								
						if(record_atendance):
							# draw the bounding box
							text = "{}: {:.2f}%".format(name, proba * 100)
							y = startY - 10 if startY - 10 > 10 else startY + 10
							cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
							cv2.putText(self.frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

							# record attendance to file
							with open('dataset/student_data.json') as studentFile:
								data = json.load(studentFile)
							for studentDict in data["student"]:
								if(studentDict["ID"] == name):
									if(name not in recorded):
										now = datetime.datetime.now()
										current_time = now.strftime("%H:%M:%S")
										student_dict = {"ID" : name, "name" : studentDict["name"], "time" : str(current_time), "subject" : subject}
										attendance_list.append(student_dict)
										self.DisplayAttendance(list(student_dict.values()))
										recorded.append(name)
						else:
							# check if teacher
							for teacherDict in data["teacher"]:

								if(teacherDict["ID"] == name):

									# draw the bounding box
									text = "{}: {:.2f}%".format(name, proba * 100)
									y = startY - 10 if startY - 10 > 10 else startY + 10
									cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
									cv2.putText(self.frame, text, (startX, y),
									cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

									print( teacherDict)
									verify = input("Record attendance(yes/no) : ")

									if(verify == "yes"):
										record_atendance = True
										subject = teacherDict["subject"]
				
				# show the output frame
				cv2.imshow("Frame", self.frame)

				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = ImagePIL.fromarray(image)
				image = ImageTk.PhotoImage(image)

				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(self.rootAttendance, image=image)
					self.panel.image = image
					self.panel.grid(row=0, column=0)
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image
				key = cv2.waitKey(1) & 0xFF

				# if the `s` key was pressed, stop attendance recording
				if key == ord("s"):
					print(attendance_list)
					attendance_dict = {subject : attendance_list}
					#db.child("attendance").child(str(date.today())).push(attendance_dict)
					record_atendance = False
					attendance_list = []
					recorded = []
					subject = ""
					
				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

			print(attendance_list)

			# do a bit of cleanup
			cv2.destroyAllWindows()
			self.vs.stop()

		except RuntimeError:
			print("[INFO] caught a RuntimeError")

	def takeSnapshot(self):
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))
		# save the file
		cv2.imwrite(p, self.frame.copy())
		print("[INFO] saved {}".format(filename))

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.rootAttendance.quit()

	def MainLoop(self):
		self.rootAttendance.mainloop()