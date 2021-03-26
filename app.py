# import packages
from sklearn.preprocessing import LabelEncoder
from env_variables import firebaseConfig
from imutils.video import VideoStream
from imutils.video import FPS
from sklearn.svm import SVC
from tkinter import messagebox
from imutils import paths
import pyrebase
from tkinter import*
import numpy as np
import imutils
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

        self.root.mainloop()

    def AddTeacherPressed(self):
        self.addTeacher = AddTeacher()
    
    def AddStudentPressed(self):
        self.addStudent = AddStudent()
    
    def StartAttendancePressed(self):
        self.startAttendance = StartAttendance()

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
        self.db.child("teacher").child(self.IDTeacher.get()).push(self.teacherDict)

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

# Start ATTENDANCE
class StartAttendance():
    def __init__(self):
        os.system('cmd /k' "python recognize_video.py --detector model --embedding-model model/openface_nn4.small2.v1.t7 --recognizer output/face_recognizer.pickle --le output/face_label.pickle")

homeScreen = HomeScreen()