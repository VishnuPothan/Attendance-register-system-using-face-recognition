# import the packages
from sklearn.preprocessing import LabelEncoder
from env_variables import firebaseConfig
from imutils.video import VideoStream
from imutils.video import FPS
from sklearn.svm import SVC
from tkinter import messagebox
from imutils import paths
from photoboothapp import PhotoBoothApp
import pyrebase
from tkinter import*
import numpy as np
import imutils
import pickle
import json
import time
import cv2
import os


#-----------------INITIALIZE TKINTER-------------------------------------------------------------------
root = Tk() 
root.title("Attendance Register Using Face Recognition")
width, height = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry('%dx%d+0+0' % (width,height))

#-----------------INFO TOP-----------------------------------------------------------------------------
Tops = Frame(root,bg="white",width = 1600,height=50,relief=SUNKEN)
Tops.pack(side=TOP)

lblInfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
lblInfo.grid(row=0,column=0)
lblInfo = Label(Tops, font=( 'aria' ,20, ),text="Group 2",fg="steel blue",anchor=W)
lblInfo.grid(row=1,column=0)

#-----------------INITIALIZE TKINTER UI ADD TEACHER-------------------------------------------------------------------
rootAddTeacher = Tk() 
rootAddTeacher.title("Attendance Register Using Face Recognition")
width, height = rootAddTeacher.winfo_screenwidth(), rootAddTeacher.winfo_screenheight()
rootAddTeacher.geometry('%dx%d+0+0' % (width,height))
rootAddTeacher.lift()

#-----------------INITIALIZE TKINTER UI ADD STUDENT-------------------------------------------------------------------
rootAddStudent = Tk() 
rootAddStudent.title("Attendance Register Using Face Recognition")
width, height = rootAddStudent.winfo_screenwidth(), rootAddStudent.winfo_screenheight()
rootAddStudent.geometry('%dx%d+0+0' % (width,height))
 
# Hide Windows
rootAddTeacher.withdraw()
rootAddStudent.withdraw()

# variables
nameTeacher = StringVar(rootAddTeacher)
IDTeacher = StringVar(rootAddTeacher)
subjectTeacher = StringVar(rootAddTeacher)

nameStudent = StringVar(rootAddStudent)
IDStudent = StringVar(rootAddStudent)

teacherUIAdded = False
studentUIAdded = False

# Function Actions
def OnClosing():
    rootAddStudent.destroy()
    rootAddTeacher.destroy()
    root.destroy()

def OnClosingAddTeacher():
    rootAddTeacher.withdraw()

def OnClosingAddStudent():
    rootAddStudent.withdraw()

def ResetTeacherWindow():
    IDTeacher.set("")
    nameTeacher.set("")
    subjectTeacher.set("")

def ResetStudentWindow():
    IDStudent.set("")
    nameStudent.set("")

def CaptureImageTeacher():
    if (len(IDTeacher.get()) == 0):
        messagebox.showerror("Error", "Enter Teacher ID To Continue")
        return
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("Face")
    img_counter = 0
    os.mkdir("dataset/"+IDTeacher.get()) 
    
    while True:
        ret, frame = cam.read()
        if not ret:
            messagebox.showwarning("Error", "failed to grab frame")
            break

        cv2.imshow("Face Capture", frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            if(img_counter > 10):
                break
            messagebox.showwarning("Warning", "Image Count Is Low, Keep Taking Images")

        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/{}/{}.png".format(IDTeacher.get(), img_counter)
            #cv2.imwrite(img_name, frame)

            cv2.imwrite(img_name, frame)

            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

def AddNewTeacher():
    if (len(IDTeacher.get()) == 0 or len(nameTeacher.get()) == 0 or len(subjectTeacher.get()) == 0):
        messagebox.showerror("Error", "Enter Teacher Details To Continue")
        return

    # Vaiable declaration
    dataset = "dataset/" + IDTeacher.get()
    embeddings = "output/face_embed.pickle"
    embeddingModel = "model/openface_nn4.small2.v1.t7"
    detector = "model"
    recognizerPickle = "output/face_recognizer.pickle"
    lePickle = "output/face_label.pickle"
    confidence = 0.5

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize lists of extracted facial embeddings and people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = IDTeacher.get()

        # load the image, resize it to width of 600 pixels (maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # assuming that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidenceDetection = detections[0, 0, i, 2]

        # filter out weak detections
        if confidenceDetection > confidence:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
    
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings, "rb").read())

    data["embeddings"] = data["embeddings"] + knownEmbeddings
    data["names"] = data["names"] + knownNames

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

        
    # write the actual face recognition model to disk
    f = open(recognizerPickle, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(lePickle, "wb")
    f.write(pickle.dumps(le))
    f.close()

    # write data to firebase
    # initlaize firebase connection
    firebase = pyrebase.initialize_app(firebaseConfig)
    db = firebase.database()

    # write to realtime db firebase
    teacherDict = {"ID" : IDTeacher.get(), "name" : nameTeacher.get(), "subject" : subjectTeacher.get()}
    db.child("teacher").child(IDTeacher.get()).push(teacherDict)

    # Reset Window
    ResetTeacherWindow()
    rootAddTeacher.withdraw()

def CaptureImageStudent():
    if (len(IDStudent.get()) == 0):
        messagebox.showerror("Error", "Enter Student ID To Continue")
        return
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("Face")
    img_counter = 0
    os.mkdir("dataset/"+IDStudent.get()) 
    
    while True:
        ret, frame = cam.read()
        if not ret:
            messagebox.showwarning("Error", "failed to grab frame")
            break

        cv2.imshow("Face Capture", frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            if(img_counter > 10):
                break
            messagebox.showwarning("Warning", "Image Count Is Low, Keep Taking Images")

        elif k%256 == 32:
            # SPACE pressed
            img_name = "dataset/{}/{}.png".format(IDStudent.get(), img_counter)
            #cv2.imwrite(img_name, frame)

            cv2.imwrite(img_name, frame)

            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

def AddNewStudent():
    if (len(IDStudent.get()) == 0 or len(nameStudent.get()) == 0):
        messagebox.showerror("Error", "Enter Student Details To Continue")
        return

    # Vaiable declaration
    dataset = "dataset/" + IDStudent.get()
    embeddings = "output/face_embed.pickle"
    embeddingModel = "model/openface_nn4.small2.v1.t7"
    detector = "model"
    recognizerPickle = "output/face_recognizer.pickle"
    lePickle = "output/face_label.pickle"
    confidence = 0.5

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize lists of extracted facial embeddings and people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = IDStudent.get()

        # load the image, resize it to width of 600 pixels (maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # assuming that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidenceDetection = detections[0, 0, i, 2]

        # filter out weak detections
        if confidenceDetection > confidence:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
    
    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings, "rb").read())

    data["embeddings"] = data["embeddings"] + knownEmbeddings
    data["names"] = data["names"] + knownNames

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

        
    # write the actual face recognition model to disk
    f = open(recognizerPickle, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(lePickle, "wb")
    f.write(pickle.dumps(le))
    f.close()

    # write data to firebase
    # initlaize firebase connection
    firebase = pyrebase.initialize_app(firebaseConfig)
    db = firebase.database()

    # write to realtime db firebase
    studentDict = {"ID" : IDStudent.get(), "name" : nameStudent.get()}
    db.child("student").child(IDStudent.get()).push(studentDict)

    # Reset Sudent Window
    ResetStudentWindow()
    rootAddStudent.withdraw()

def StartAttendance():
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
    pba = PhotoBoothApp(vs, "output", detector, embedder, recognizer, le)
    pba.MainLoop()

# Funtions UI
def UIAddTeacher():
    rootAddTeacher.deiconify()
    rootAddTeacher.lift()
    
    global teacherUIAdded
    if(teacherUIAdded):
        return
    teacherUIAdded = True

    #-----------------INFO TOP-----------------------------------------------------------------------------
    Tops = Frame(rootAddTeacher,bg="white",width = 1600,height=50,relief=SUNKEN)
    Tops.pack(side=TOP)

    lblInfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
    lblInfo.grid(row=0,column=0)
    lblInfo = Label(Tops, font=( 'aria' ,20, ),text="Add Teacher",fg="steel blue",anchor=W)
    lblInfo.grid(row=1,column=0)

    #-----------------FORM ITEMS--------------------------------------------------------------------------------------------
    f1 = Frame(rootAddTeacher,width = 900,height=700,relief=SUNKEN)
    f1.pack(side=TOP)

    # Name input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=0,columnspan=3)

    lblName = Label(f1, font=( 'aria' ,16, 'bold' ),text="Name",fg="steel blue",bd=10,anchor='w')
    lblName.grid(row=1,column=0)

    txtName = Entry(f1,font=('ariel' ,16,'bold'), textvariable=nameTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
    txtName.grid(row=1,column=1)

    # ID input    
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=2,columnspan=3)

    lblID = Label(f1, font=( 'aria' ,16, 'bold' ),text="ID",fg="steel blue",bd=10,anchor='w')
    lblID.grid(row=3,column=0)

    txtID = Entry(f1,font=('ariel' ,16,'bold'), textvariable=IDTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
    txtID.grid(row=3,column=1)

    # Subject input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=4,columnspan=3)

    lblSubject = Label(f1, font=( 'aria' ,16, 'bold' ),text="Subject",fg="steel blue",bd=10,anchor='w')
    lblSubject.grid(row=5,column=0)

    txtSubject = Entry(f1,font=('ariel' ,16,'bold'), textvariable=subjectTeacher , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
    txtSubject.grid(row=5,column=1)

    #  Face input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=6,columnspan=3)

    lblFace = Label(f1, font=( 'aria' ,16, 'bold' ),text="Face",fg="steel blue",bd=10,anchor='w')
    lblFace.grid(row=7,column=0)

    btnFace = Button(f1,padx=16,pady=8, bd=6 ,fg="black",font=('ariel' ,16,'bold'),width=4, height=1, text="Capture", bg="grey",command = CaptureImageTeacher)
    btnFace.grid(row=7, column=1)

    # Add Button
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=8,columnspan=3)

    btnStart = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add", bg="powder blue",command = AddNewTeacher)
    btnStart.grid(row=9, column=1)

    rootAddTeacher.mainloop()

def UIAddStudent():
    rootAddStudent.deiconify()
    rootAddStudent.lift()

    global studentUIAdded
    if(studentUIAdded):
        return
    studentUIAdded = True

    #-----------------INFO TOP-----------------------------------------------------------------------------
    Tops = Frame(rootAddStudent,bg="white",width = 1600,height=50,relief=SUNKEN)
    Tops.pack(side=TOP)

    lblInfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Attendance Register Using Face Recognition",fg="steel blue",bd=10,anchor='w')
    lblInfo.grid(row=0,column=0)

    lblInfo = Label(Tops, font=( 'aria' ,20, ),text="Add Student",fg="steel blue",anchor=W)
    lblInfo.grid(row=1,column=0)
    #-----------------FORM ITEMS---------------------------------------------------------------------------
    f1 = Frame(rootAddStudent,width = 900,height=700,relief=SUNKEN)
    f1.pack(side=TOP)

    # Name Input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=0,columnspan=3)  

    lblName = Label(f1, font=( 'aria' ,16, 'bold' ),text="Name",fg="steel blue",bd=10,anchor='w')
    lblName.grid(row=1,column=0)

    txtName = Entry(f1,font=('ariel' ,16,'bold'), textvariable=nameStudent , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
    txtName.grid(row=1,column=1)

    # ID Input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=2,columnspan=3)

    lblID = Label(f1, font=( 'aria' ,16, 'bold' ),text="ID",fg="steel blue",bd=10,anchor='w')
    lblID.grid(row=3,column=0)

    txtID = Entry(f1,font=('ariel' ,16,'bold'), textvariable=IDStudent , bd=6,insertwidth=4,bg="powder blue" ,justify='right')
    txtID.grid(row=3,column=1) 

    #  Face input
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=4,columnspan=3)

    lblFace = Label(f1, font=( 'aria' ,16, 'bold' ),text="Face",fg="steel blue",bd=10,anchor='w')
    lblFace.grid(row=5,column=0)

    btnFace = Button(f1,padx=16,pady=8, bd=6 ,fg="black",font=('ariel' ,16,'bold'),width=4, height=1, text="Capture", bg="grey",command = CaptureImageStudent)
    btnFace.grid(row=5, column=1)

    # Add Button
    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=6,columnspan=3)

    btnStart = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add", bg="powder blue",command=AddNewStudent)
    btnStart.grid(row=7, column=1)

    rootAddStudent.mainloop()

def MainFooter():
    #-----------------MENU ITEMS---------------------------------------------------------------------------
    f1 = Frame(root,width = 900,height=700,relief=SUNKEN)
    f1.pack(side=TOP)

    lblDash = Label(f1,text="---------------------",fg="black")
    lblDash.grid(row=6,columnspan=3)

    btnStart = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Start Attendance", bg="powder blue",command=StartAttendance)
    btnStart.grid(row=8, column=1)

    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=9,columnspan=3)

    btnAddTeacher = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add Teacher", bg="powder blue",command=UIAddTeacher)
    btnAddTeacher.grid(row=10, column=1)

    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=11,columnspan=3)

    btnAddStudent = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="Add Student", bg="powder blue",command=UIAddStudent)
    btnAddStudent.grid(row=12, column=1)

    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=13,columnspan=3)

    btnViewTeacher= Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="View Teachers", bg="powder blue",command=None)
    btnViewTeacher.grid(row=14, column=1)

    lblDash = Label(f1,text="---------------------",fg="white")
    lblDash.grid(row=15,columnspan=3)

    btnViewStudent = Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=20, text="View Students", bg="powder blue",command=None)
    btnViewStudent.grid(row=16, column=1)

    root.mainloop()

rootAddStudent.protocol("WM_DELETE_WINDOW", OnClosingAddStudent)
rootAddTeacher.protocol("WM_DELETE_WINDOW", OnClosingAddTeacher)
root.protocol("WM_DELETE_WINDOW", OnClosing)

MainFooter()
