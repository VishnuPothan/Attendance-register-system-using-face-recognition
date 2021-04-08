# import packages
from sklearn.preprocessing import LabelEncoder
from env_variables import firebaseConfig
from imutils.video import VideoStream
from PIL import  Image as Img
from PIL import ImageTk
from imutils.video import FPS
from sklearn.svm import SVC
from tkinter import messagebox
from imutils import paths
from tkinter import Tk, Label
import pyrebase
from tkinter import *
import tkinter as tki
import numpy as np
import imutils
import datetime
import pickle
import json
import time
import cv2
import os

# HOME SCREEN
class HomeScreen():
    def __init__(self):
        #-----------------INITIALIZE TKINTER-------------------------------------------------------------------
        self.root = Tk() 
        self.root.title("Attendance Register Using Face Recognition")
        self.width, self.height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry('%dx%d+0+0' % (self.width,self.height))

        #-----------------INFO TOP-----------------------------------------------------------------------------
        self.Tops = Frame(self.root,bg="white",width = 1600,height=50,relief=SUNKEN)
        self.Tops.pack(side=TOP)

        self.lblInfo = Label(self.Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
        self.lblInfo.grid(row=0,column=0)
        self.lblInfo = Label(self.Tops, font=( 'aria' ,20, ),text="Group 2",fg="steel blue",anchor=W)
        self.lblInfo.grid(row=1,column=0)
        
        #-----------------MENU ITEMS---------------------------------------------------------------------------
        self.f1 = Frame(self.root,width = 900,height=700,relief=SUNKEN)
        self.f1.pack(side=TOP)

        self.lblDash = Label(self.f1,text="---------------------",fg="black")
        self.lblDash.grid(row=6,columnspan=3)

        self.btnStart = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Start Attendance", bg="powder blue",command = self.StartAttendancePressed )
        self.btnStart.grid(row=8, column=1)

        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=9,columnspan=3)

        self.btnAddTeacher = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add Teacher", bg="powder blue",command = self.AddTeacherPressed)
        self.btnAddTeacher.grid(row=10, column=1)

        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=11,columnspan=3)

        self.btnAddStudent = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add Student", bg="powder blue",command = self.AddStudentPressed)
        self.btnAddStudent.grid(row=12, column=1)

        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=13,columnspan=3)

        self.btnViewTeacher= Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="View Teachers", bg="powder blue",command=None)
        self.btnViewTeacher.grid(row=14, column=1)

        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=15,columnspan=3)

        self.btnViewStudent = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="View Students", bg="powder blue",command=None)
        self.btnViewStudent.grid(row=16, column=1)

        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=17,columnspan=3)

        self.btnAddClass = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add Class", bg="powder blue",command = self.AddClassPressed)
        self.btnAddClass.grid(row=18, column=1)

        self.root.mainloop()

    def AddTeacherPressed(self):
        self.addTeacher = AddTeacher()
    
    def AddStudentPressed(self):
        self.addStudent = AddStudent()
    
    def StartAttendancePressed(self):
        self.startAttendance = StartAttendance()

    def AddClassPressed(self):
        self.addClass = AddClass()

# ADD TEACHER SCREEN
class AddTeacher():
    def __init__(self):

        #-----------------INITIALIZE TKINTER UI ADD TEACHER-------------------------------------------------------------------
        self.rootAddTeacher = Tk() 
        self.rootAddTeacher.title("Attendance Register Using Face Recognition")
        self.width, self.height = self.rootAddTeacher.winfo_screenwidth(), self.rootAddTeacher.winfo_screenheight()
        self.rootAddTeacher.geometry('%dx%d+0+0' % (self.width,self.height))
        self.rootAddTeacher.lift()

        # variables
        self.nameTeacher = StringVar(self.rootAddTeacher)
        self.IDTeacher = StringVar(self.rootAddTeacher)
        self.subjectTeacher = StringVar(self.rootAddTeacher)

        #-----------------INFO TOP-----------------------------------------------------------------------------
        self.Tops = Frame(self.rootAddTeacher,bg="white",width = 1600,height=50,relief=SUNKEN)
        self.Tops.pack(side=TOP)

        self.lblInfo = Label(self.Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
        self.lblInfo.grid(row=0,column=0)
        self.lblInfo = Label(self.Tops, font=( 'aria' ,20, ),text="Add Teacher",fg="steel blue",anchor=W)
        self.lblInfo.grid(row=1,column=0)

        #-----------------FORM ITEMS--------------------------------------------------------------------------------------------
        self.f1 = Frame(self.rootAddTeacher,width = 900,height=700,relief=SUNKEN)
        self.f1.pack(side=TOP)

        # Name input
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=0,columnspan=3)

        self.lblName = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Name",fg="steel blue",bd=10,anchor='w')
        self.lblName.grid(row=1,column=0)

        self.txtName = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.nameTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtName.grid(row=1,column=1)

        # ID input    
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=2,columnspan=3)

        self.lblID = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="ID",fg="steel blue",bd=10,anchor='w')
        self.lblID.grid(row=3,column=0)

        self.txtID = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.IDTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtID.grid(row=3,column=1)

        # Subject input
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=4,columnspan=3)

        self.lblSubject = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Subject",fg="steel blue",bd=10,anchor='w')
        self.lblSubject.grid(row=5,column=0)

        self.txtSubject = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.subjectTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtSubject.grid(row=5,column=1)

        #  Face input 
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=6,columnspan=3)

        self.lblFace = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Face",fg="steel blue",bd=10,anchor='w')
        self.lblFace.grid(row=7,column=0)

        self.btnFace = Button(self.f1,padx=16,pady=8, bd=6 ,fg="black",font=('ariel' ,16,'bold'),width=4, height=1, text="Capture", bg="grey",command = self.CaptureImageTeacher)
        self.btnFace.grid(row=7, column=1)

        # Add Button
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=8,columnspan=3)

        self.btnStart = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add", bg="powder blue",command = self.AddNewTeacher)
        self.btnStart.grid(row=9, column=1)

        self.rootAddTeacher.mainloop()
    
    def CaptureImageTeacher(self):
        # Show error on No ID Teacher No Entered 
        if (len(self.IDTeacher.get()) == 0):
            messagebox.showerror("Error", "Enter Teacher ID To Continue")
            return
    
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Face")
        self.img_counter = 0
        os.mkdir("dataset/"+self.IDTeacher.get()) 
        
        while True:
            self.ret, self.frame = self.cam.read()
            if not self.ret:
                messagebox.showwarning("Error", "failed to grab frame")
                break

            cv2.imshow("Face Capture", self.frame)
            self.k = cv2.waitKey(1)

            if self.k%256 == 27:
                # ESC pressed
                if(self.img_counter > 10):
                    break
                self.messagebox.showwarning("Warning", "Image Count Is Low, Keep Taking Images")

            elif self.k%256 == 32:
                # SPACE pressed
                self.img_name = "dataset/{}/{}.png".format(self.IDTeacher.get(), self.img_counter)
                #cv2.imwrite(img_name, frame)

                cv2.imwrite(self.img_name, self.frame)

                print("{} written!".format(self.img_name))
                self.img_counter += 1

        self.cam.release()
        cv2.destroyAllWindows()

    def AddNewTeacher(self):
        if (len(self.IDTeacher.get()) == 0 or len(self.nameTeacher.get()) == 0 or len(self.subjectTeacher.get()) == 0):
            messagebox.showerror("Error", "Enter Teacher Details To Continue")
            return

        # Vaiable declaration
        self.dataset = "dataset/" + self.IDTeacher.get()
        self.embeddings = "output/face_embed.pickle"
        self.embeddingModel = "model/openface_nn4.small2.v1.t7"
        self.detector = "model"
        self.recognizerPickle = "output/face_recognizer.pickle"
        self.lePickle = "output/face_label.pickle"
        self.confidence = 0.5

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        self.protoPath = os.path.sep.join([self.detector, "deploy.prototxt"])
        self.modelPath = os.path.sep.join([self.detector, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.embeddingModel)

        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        self.imagePaths = list(paths.list_images(self.dataset))

        # initialize lists of extracted facial embeddings and people names
        self.knownEmbeddings = []
        self.knownNames = []

        # initialize the total number of faces processed
        self.total = 0

        # loop over the image paths
        for (self.i, self.imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(self.i + 1, len(self.imagePaths)))
            self.name = self.IDTeacher.get()

            # load the image, resize it to width of 600 pixels (maintaining the aspect ratio), and then grab the image dimensions
            self.image = cv2.imread(self.imagePath)
            self.image = imutils.resize(self.image, width=600)
            (self.h, self.w) = self.image.shape[:2]

            # construct a blob from the image
            self.imageBlob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize faces in the input image
            self.detector.setInput(self.imageBlob)
            self.detections = self.detector.forward()

            # ensure at least one face was found
            if len(self.detections) > 0:
                # assuming that each image has only ONE face, so find the bounding box with the largest probability
                self.i = np.argmax(self.detections[0, 0, :, 2])
                self.confidenceDetection = self.detections[0, 0, self.i, 2]

            # filter out weak detections
            if self.confidenceDetection > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the face
                self.box = self.detections[0, 0, self.i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                self.face = self.image[self.startY:self.endY, self.startX:self.endX]
                (self.fH, self.fW) = self.face.shape[:2]

                # ensure the face width and height are sufficiently large
                if self.fW < 20 or self.fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                self.faceBlob = cv2.dnn.blobFromImage(self.face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(self.faceBlob)
                self.vec = self.embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                self.knownNames.append(self.name)
                self.knownEmbeddings.append(self.vec.flatten())
                self.total += 1
        
        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(self.total))

        # load the face embeddings
        print("[INFO] loading face embeddings...")
        self.data = pickle.loads(open(self.embeddings, "rb").read())

        self.data["embeddings"] = self.data["embeddings"] + self.knownEmbeddings
        self.data["names"] = self.data["names"] + self.knownNames

        # encode the labels
        print("[INFO] encoding labels...")
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.data["names"])

        # train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
        print("[INFO] training model...")
        self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
        self.recognizer.fit(self.data["embeddings"], self.labels)

            
        # write the actual face recognition model to disk
        self.f = open(self.recognizerPickle, "wb")
        self.f.write(pickle.dumps(self.recognizer))
        self.f.close()

        # write the label encoder to disk
        self.f = open(self.lePickle, "wb")
        self.f.write(pickle.dumps(self.le))
        self.f.close()

        # write data to firebase
        # initlaize firebase connection
        self.firebase = pyrebase.initialize_app(firebaseConfig)
        self.db = self.firebase.database()

        # write to realtime db firebase
        self.teacherDict = {"ID" : self.IDTeacher.get(), "name" : self.nameTeacher.get(), "subject" : self.subjectTeacher.get()}
        self.db.child("class").child("S8CSE").child("teacher").child(self.IDTeacher.get()).set(self.teacherDict)

        # Reset Window
        #ResetTeacherWindow()
        self.rootAddTeacher.destroy()

# ADD STUDENT SCREEN
class AddStudent():
    def __init__(self):

        #-----------------INITIALIZE TKINTER UI ADD STUDENT-------------------------------------------------------------------
        self.rootAddStudent = Tk() 
        self.rootAddStudent.title("Attendance Register Using Face Recognition")
        self.width, self.height = self.rootAddStudent.winfo_screenwidth(), self.rootAddStudent.winfo_screenheight()
        self.rootAddStudent.geometry('%dx%d+0+0' % (self.width,self.height))
        self.rootAddStudent.lift()

        # variables
        self.nameStudent = StringVar(self.rootAddStudent)
        self.IDStudent = StringVar(self.rootAddStudent)

        #-----------------INFO TOP-----------------------------------------------------------------------------
        self.Tops = Frame(self.rootAddStudent,bg="white",width = 1600,height=50,relief=SUNKEN)
        self.Tops.pack(side=TOP)

        self.lblInfo = Label(self.Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
        self.lblInfo.grid(row=0,column=0)
        self.lblInfo = Label(self.Tops, font=( 'aria' ,20, ),text="Add Student",fg="steel blue",anchor=W)
        self.lblInfo.grid(row=1,column=0)

        #-----------------FORM ITEMS--------------------------------------------------------------------------------------------
        self.f1 = Frame(self.rootAddStudent,width = 900,height=700,relief=SUNKEN)
        self.f1.pack(side=TOP)

        # Name input
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=0,columnspan=3)

        self.lblName = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Name",fg="steel blue",bd=10,anchor='w')
        self.lblName.grid(row=1,column=0)

        self.txtName = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.nameStudent , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtName.grid(row=1,column=1)

        # ID input    
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=2,columnspan=3)

        self.lblID = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="ID",fg="steel blue",bd=10,anchor='w')
        self.lblID.grid(row=3,column=0)

        self.txtID = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.IDStudent , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtID.grid(row=3,column=1)

        #  Face input 
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=6,columnspan=3)

        self.lblFace = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Face",fg="steel blue",bd=10,anchor='w')
        self.lblFace.grid(row=7,column=0)

        self.btnFace = Button(self.f1,padx=16,pady=8, bd=6 ,fg="black",font=('ariel' ,16,'bold'),width=4, height=1, text="Capture", bg="grey",command = self.CaptureImageStudent)
        self.btnFace.grid(row=7, column=1)

        # Add Button
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=8,columnspan=3)

        self.btnStart = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add", bg="powder blue",command = self.AddNewStudent)
        self.btnStart.grid(row=9, column=1)

        self.rootAddStudent.mainloop()
    
    def CaptureImageStudent(self):
        # Show error on No ID Student No Entered 
        if (len(self.IDStudent.get()) == 0):
            messagebox.showerror("Error", "Enter Student ID To Continue")
            return
    
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.namedWindow("Face")
        self.img_counter = 0
        os.mkdir("dataset/"+self.IDStudent.get()) 
        
        while True:
            self.ret, self.frame = self.cam.read()
            if not self.ret:
                messagebox.showwarning("Error", "failed to grab frame")
                break

            cv2.imshow("Face Capture", self.frame)
            self.k = cv2.waitKey(1)

            if self.k%256 == 27:
                # ESC pressed
                if(self.img_counter > 10):
                    break
                self.messagebox.showwarning("Warning", "Image Count Is Low, Keep Taking Images")

            elif self.k%256 == 32:
                # SPACE pressed
                self.img_name = "dataset/{}/{}.png".format(self.IDStudent.get(), self.img_counter)
                #cv2.imwrite(img_name, frame)

                cv2.imwrite(self.img_name, self.frame)

                print("{} written!".format(self.img_name))
                self.img_counter += 1

        self.cam.release()
        cv2.destroyAllWindows()

    def AddNewStudent(self):
        if (len(self.IDStudent.get()) == 0 or len(self.nameStudent.get()) == 0):
            messagebox.showerror("Error", "Enter Student Details To Continue")
            return

        # Vaiable declaration
        self.dataset = "dataset/" + self.IDStudent.get()
        self.embeddings = "output/face_embed.pickle"
        self.embeddingModel = "model/openface_nn4.small2.v1.t7"
        self.detector = "model"
        self.recognizerPickle = "output/face_recognizer.pickle"
        self.lePickle = "output/face_label.pickle"
        self.confidence = 0.5

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        self.protoPath = os.path.sep.join([self.detector, "deploy.prototxt"])
        self.modelPath = os.path.sep.join([self.detector, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.embeddingModel)

        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        self.imagePaths = list(paths.list_images(self.dataset))

        # initialize lists of extracted facial embeddings and people names
        self.knownEmbeddings = []
        self.knownNames = []

        # initialize the total number of faces processed
        self.total = 0

        # loop over the image paths
        for (self.i, self.imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(self.i + 1, len(self.imagePaths)))
            self.name = self.IDStudent.get()

            # load the image, resize it to width of 600 pixels (maintaining the aspect ratio), and then grab the image dimensions
            self.image = cv2.imread(self.imagePath)
            self.image = imutils.resize(self.image, width=600)
            (self.h, self.w) = self.image.shape[:2]

            # construct a blob from the image
            self.imageBlob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize faces in the input image
            self.detector.setInput(self.imageBlob)
            self.detections = self.detector.forward()

            # ensure at least one face was found
            if len(self.detections) > 0:
                # assuming that each image has only ONE face, so find the bounding box with the largest probability
                self.i = np.argmax(self.detections[0, 0, :, 2])
                self.confidenceDetection = self.detections[0, 0, self.i, 2]

            # filter out weak detections
            if self.confidenceDetection > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the face
                self.box = self.detections[0, 0, self.i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                self.face = self.image[self.startY:self.endY, self.startX:self.endX]
                (self.fH, self.fW) = self.face.shape[:2]

                # ensure the face width and height are sufficiently large
                if self.fW < 20 or self.fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                self.faceBlob = cv2.dnn.blobFromImage(self.face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(self.faceBlob)
                self.vec = self.embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                self.knownNames.append(self.name)
                self.knownEmbeddings.append(self.vec.flatten())
                self.total += 1
        
        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(self.total))

        # load the face embeddings
        print("[INFO] loading face embeddings...")
        self.data = pickle.loads(open(self.embeddings, "rb").read())

        self.data["embeddings"] = self.data["embeddings"] + self.knownEmbeddings
        self.data["names"] = self.data["names"] + self.knownNames

        # encode the labels
        print("[INFO] encoding labels...")
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.data["names"])

        # train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
        print("[INFO] training model...")
        self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
        self.recognizer.fit(self.data["embeddings"], self.labels)

            
        # write the actual face recognition model to disk
        self.f = open(self.recognizerPickle, "wb")
        self.f.write(pickle.dumps(self.recognizer))
        self.f.close()

        # write the label encoder to disk
        self.f = open(self.lePickle, "wb")
        self.f.write(pickle.dumps(self.le))
        self.f.close()

        # write data to firebase
        # initlaize firebase connection
        self.firebase = pyrebase.initialize_app(firebaseConfig)
        self.db = self.firebase.database()

        # write to realtime db firebase
        self.studentDict = {"ID" : self.IDStudent.get(), "name" : self.nameStudent.get()}
        self.db.child("student").child(self.IDStudent.get()).push(self.studentDict)

        # Reset Window
        #ResetStudentWindow()
        self.rootAddStudent.destroy()

# START ATTENDANCE
class StartAttendance():
    def __init__(self):
        os.system('cmd /k' "python recognize_video.py --detector model --embedding-model model/openface_nn4.small2.v1.t7 --recognizer output/face_recognizer.pickle --le output/face_label.pickle")
                
        #-----------------INITIALIZE TKINTER UI ADD STUDENT-------------------------------------------------------------------
        """ self.rootAttendance = tki.Toplevel()
        self.rootAttendance.title("Attendance Register Using Face Recognition")
        self.width, self.height = self.rootAttendance.winfo_screenwidth(), self.rootAttendance.winfo_screenheight()
        self.rootAttendance.geometry('%dx%d+0+0' % (self.width,self.height))

        Label(self.rootAttendance,  font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w').pack()
        self.f1 = LabelFrame(self.rootAttendance,bg='red')

        self.lblInfo = Label(self.rootAttendance, font=( 'aria' ,20, ),text="Attendance Record",fg="steel blue",anchor=W).pack()

        self.f1.pack()
        self.L1 = Label(self.f1, bg='powder blue')
        self.L1.pack()
        self.cap = VideoStream().start()

        self.detectorPath = "model"
        self.embeddingModelPath = "model/openface_nn4.small2.v1.t7"
        self.recognizerPath = "output/face_recognizer.pickle"
        self.labelPath = "output/face_label.pickle"

        self.detectorPath = "model"
        self.embeddingModelPath = "model/openface_nn4.small2.v1.t7"
        self.recognizerPath = "output/face_recognizer.pickle"
        self.labelPath = "output/face_label.pickle"

        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        self.protoPath = os.path.sep.join([self.detectorPath, "deploy.prototxt"])
        self.modelPath = os.path.sep.join([self.detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.embeddingModelPath)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(self.recognizerPath, "rb").read())
        self.le = pickle.loads(open(self.labelPath, "rb").read())

        # teacher found
        self.record_atendance = False
        self.attendance_list = []
        self.recorded = []
        self.subject = ""

        # Loope Video frames until stop condition
        while True:
            # Read Frame
            self.frame = self.cap.read()
            (self.h, self.w) = self.frame.shape[:2]
            self.frame = imutils.resize(self.frame, width=300)
            self.img = self.cap.read()

            # Image conver to RGB
            self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.image = Img.fromarray(self.image)
            self.image = ImageTk.PhotoImage(self.image)

            # construct a blob from the image
            self.imageBlob = cv2.dnn.blobFromImage(cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # faces in the input image
            self.detector.setInput(self.imageBlob)
            self.detections = self.detector.forward()

            # loop over the detections
            for self.i in range(0, self.detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                self.confidence = self.detections[0, 0, self.i, 2]

                # filter out weak detections
                if self.confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    self.box = self.detections[0, 0, self.i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                    (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

                    # extract the face ROI
                    self.face = self.frame[self.startY:self.endY, self.startX:self.endX]
                    (self.fH, self.fW) = self.face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if self.fW < 20 or self.fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    self.faceBlob = cv2.dnn.blobFromImage(self.face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(self.faceBlob)
                    self.vec = self.embedder.forward()

                    # perform classification to recognize the face
                    self.preds = self.recognizer.predict_proba(self.vec)[0]
                    self.j = np.argmax(self.preds)
                    self.proba = self.preds[self.j]
                    self.name = self.le.classes_[self.j]

                    # teacher face check
                    with open('dataset/teacher_data.json') as self.teacherFile:
                        self.data = json.load(self.teacherFile)
                            
                    if(self.record_atendance):
                        # draw the bounding box
                        self.text = "{}: {:.2f}%".format(self.name, self.proba * 100)
                        self.y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10
                        cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY), (0, 0, 255), 2)
                        cv2.putText(self.frame, self.text, (self.startX, self.y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                        # record attendance to file
                        with open('dataset/student_data.json') as self.studentFile:
                            self.data = json.load(self.studentFile)
                        for self.studentDict in self.data["student"]:
                            if(self.studentDict["ID"] == self.name):
                                if(self.name not in self.recorded):
                                    self.now = datetime.datetime.now()
                                    self.current_time = self.now.strftime("%H:%M:%S")
                                    self.student_dict = {"ID" : self.name, "name" : self.studentDict["name"], "time" : str(self.current_time), "subject" : self.subject}
                                    self.attendance_list.append(self.student_dict)
                                    self.recorded.append(self.name)
                    else:
                        # check if teacher
                        for self.teacherDict in self.data["teacher"]:

                            if(self.teacherDict["ID"] == self.name):

                                # draw the bounding box
                                self.text = "{}: {:.2f}%".format(self.name, self.proba * 100)
                                self.y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10
                                cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY), (0, 0, 255), 2)
                                cv2.putText(self.frame, self.text, (self.startX, self.y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                                print(self.teacherDict)

                                self.verify = input("Record attendance(yes/no) : ")

                                if(self.verify == "yes"):
                                    self.record_atendance = True
                                    self.subject = self.teacherDict["subject"]

            print(self.attendance_list);

            # Update tikter screen
            if self.L1 is None:
                self.L1 = tki.Label(self.f1, bg='steel blue', image = self.image)
                self.L1.image = self.image
                self.L1.pack()
            else:
                self.L1.configure(image=self.image)
                self.L1.image = self.img
            self.rootAttendance.update()

            # show the output frame
            cv2.imshow("Frame", self.frame)

            self.key = cv2.waitKey(1) & 0xFF

            # if the `s` key was pressed, stop attendance recording
            if self.key == ord("s"):
                print(self.attendance_list)
                attendance_dict = {self.subject : self.attendance_list}
                #db.child("attendance").child(str(datetime.date.today())).push(attendance_dict)
                self.record_atendance = False
                self.attendance_list = []
                self.recorded = []
                self.subject = ""
                
            # if the `q` key was pressed, break from the loop
            if self.key == ord("q"):
                break

        print(self.attendance_list)
            
        # do a bit of cleanup
        self.cap.stream.release()
        cv2.destroyAllWindows() """

# ADD CLASS
class AddClass():
    def __init__(self):
        #-----------------INITIALIZE TKINTER UI ADD TEACHER-------------------------------------------------------------------
        self.rootAddClass = Tk() 
        self.rootAddClass.title("Attendance Register Using Face Recognition")
        self.width, self.height = self.rootAddClass.winfo_screenwidth(), self.rootAddClass.winfo_screenheight()
        self.rootAddClass.geometry('%dx%d+0+0' % (self.width,self.height))
        self.rootAddClass.lift()

        # variables
        self.nameClass = StringVar(self.rootAddClass)
        self.lateTime = StringVar(self.rootAddClass)
        self.hoursPerDay = StringVar(self.rootAddClass)

        #-----------------INFO TOP-----------------------------------------------------------------------------
        self.Tops = Frame(self.rootAddClass,bg="white",width = 1600,height=50,relief=SUNKEN)
        self.Tops.pack(side=TOP)

        self.lblInfo = Label(self.Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
        self.lblInfo.grid(row=0,column=0)
        self.lblInfo = Label(self.Tops, font=( 'aria' ,20, ),text="Add Teacher",fg="steel blue",anchor=W)
        self.lblInfo.grid(row=1,column=0)

        #-----------------FORM ITEMS--------------------------------------------------------------------------------------------
        self.f1 = Frame(self.rootAddClass,width = 900,height=700,relief=SUNKEN)
        self.f1.pack(side=TOP)

        # Name input
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=0,columnspan=3)

        self.lblName = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Class Name",fg="steel blue",bd=10,anchor='w')
        self.lblName.grid(row=1,column=0)

        self.txtName = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.nameClass , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtName.grid(row=1,column=1)

        # Late input    
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=2,columnspan=3)

        self.lblLateTime = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Late Time",fg="steel blue",bd=10,anchor='w')
        self.lblLateTime.grid(row=3,column=0)

        self.txtLateTime = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.lateTime , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtLateTime.grid(row=3,column=1)

        # Hours input
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=4,columnspan=3)

        self.lblHoursPerDat = Label(self.f1, font=( 'aria' ,16, 'bold' ),text="Hours Per Day",fg="steel blue",bd=10,anchor='w')
        self.lblHoursPerDat.grid(row=5,column=0)

        self.txtHoursPerDat = Entry(self.f1,font=('ariel' ,16,'bold'), textvariable = self.hoursPerDay , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
        self.txtHoursPerDat.grid(row=5,column=1)

        # Add Button
        self.lblDash = Label(self.f1,text="---------------------",fg="white")
        self.lblDash.grid(row=8,columnspan=3)

        self.btnStart = Button(self.f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add", bg="powder blue",command = self.AddNewClass)
        self.btnStart.grid(row=9, column=1)

        self.rootAddClass.mainloop()

    def AddNewClass(self):
        if (len(self.nameClass.get()) == 0 or len(self.lateTime.get()) == 0 or len(self.hoursPerDay.get()) == 0):
            messagebox.showerror("Error", "Enter Class Details To Continue")
            return
        
        # write data to firebase
        # initlaize firebase connection
        self.firebase = pyrebase.initialize_app(firebaseConfig)
        self.db = self.firebase.database()

        # write to realtime db firebase
        self.classDict = {"name" : self.nameClass.get(), "late" : self.lateTime.get(), 'hours' : self.hoursPerDay.get()}
        self.db.child("class").child(self.nameClass.get()).set(self.classDict)

        # Reset Window
        self.rootAddClass.destroy()

# CALL MAIN
homeScreen = HomeScreen()