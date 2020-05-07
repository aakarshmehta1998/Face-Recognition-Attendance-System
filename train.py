# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import *
from tkinter import Message ,Text
import shutil
import csv
import cv2,os
import numpy as np
from PIL import Image 
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
window.geometry('1050x700')
window.configure(background='grey')


#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#path = "profile.jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
#panel = tk.Label(window, image = img)


#panel.pack(side = "left", fill = "y", expand = "no")

#cv_img = cv2.imread("img541.jpg")
#x, y, no_channels = cv_img.shape
#canvas = tk.Canvas(window, width = x, height =y)
#canvas.pack(side="left")
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img)) 
# Add a PhotoImage to the Canvas
#canvas.create_image(0, 0, image=photo, anchor=tk.NW)

#msg = Message(window, text='Hello, world!')

# Font is a tuple of (font_family, size_in_points, style_modifier_string)



message = tk.Label(window, text="Face Recognition Attendance System" ,bg="black"  ,fg="white"  ,width=42  ,height=2,font=('bold', 28)) 

message.place(x=70, y=20)

lbl = tk.Label(window, text="Enter ID",width=14  ,height=2  ,fg="black"  ,bg="white" ,font=('Courier', 15, ' bold ') ) 
lbl.place(x=120, y=150)

txt = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=400, y=155)

lbl2 = tk.Label(window, text="Enter Name",width=14  ,fg="black"  ,bg="white"    ,height=2 ,font=('Courier', 15, ' bold ')) 
lbl2.place(x=120, y=250)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=400, y=260)

lbl3 = tk.Label(window, text="Notification",width=14  ,fg="black"  ,bg="white"  ,height=2 ,font=('Courier', 15, ' bold ')) 
lbl3.place(x=120, y=340)

message = tk.Label(window, text="" ,bg="white"  ,fg="black"  ,width=32  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
message.place(x=400, y=340)

lbl3 = tk.Label(window, text="Attendance",width=14  ,fg="black"  ,bg="white"  ,height=2 ,font=('Courier', 15, ' bold ')) 
lbl3.place(x=120, y=430)


message2 = tk.Label(window, text="" ,fg="black"   ,bg="white",activeforeground = "green",width=20  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=400, y=430)
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    recognizer.read("./recognizers/face-trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImage\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="white"  ,width=9  ,height=1 ,activebackground = "yellow" ,font=('Courier', 15, ' bold '))
clearButton.place(x=650, y=150)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="black"  ,bg="white"  ,width=9  ,height=1, activebackground = "yellow" ,font=('Courier', 15, ' bold '))
clearButton2.place(x=650, y=250)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="black"  ,bg="orange"  ,width=12  ,height=2, activebackground = "Red" ,font=('Helvetica', 15, ' bold '))
takeImg.place(x=120, y=530)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="black"  ,bg="orange"  ,width=12  ,height=2, activebackground = "Red" ,font=('Helvetica', 15, ' bold '))
trainImg.place(x=345, y=530)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="black"  ,bg="orange"  ,width=12  ,height=2, activebackground = "Red" ,font=('Helvetica', 15, ' bold '))
trackImg.place(x=600, y=530)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="black"  ,bg="orange"  ,width=12 ,height=2, activebackground = "Red" ,font=('Helvetica', 15, ' bold '))
quitWindow.place(x=850, y=530)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('Courier', 16, 'bold '))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Aakarsh & Team", "superscript")
copyWrite.configure(state="disabled",fg="black"  )
copyWrite.pack(side="left")
copyWrite.place(x=650, y=640)
 
window.mainloop()
