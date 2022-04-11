import numpy as np
import argparse
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import Frame,Label,Tk,Button,PhotoImage,messagebox,CENTER,TOP,BOTTOM,X,FLAT,SOLID, simpledialog
from PIL import ImageTk,Image
import pyttsx3
from threading import Thread
import imghdr
import smtplib
from email.message import EmailMessage

from imutils import face_utils

counter = 0
SENDER_EMAIL = 'fypdriverdrowsiness@gmail.com'
SENDER_PASSWORD = 'FYPdriverdrowsiness'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"Shared\shape_predictor_68_face_landmarks.dat")
model_eye = tf.keras.models.load_model(r'Eye Detection Codes\models\EyeGray Acc - 0.914 Loss - 0.266.h5')
model_yawn = tf.keras.models.load_model(r"Yawn Detection Codes\models\YawnGray Acc - 0.957 Loss - 0.117.h5") 

def mute_music():
    global muted
    
    if muted:
        btn_volume.configure(image=volume_photo)
        muted=False
    else:
        btn_volume.configure(image=mute_photo)
        muted=True

def play_warning():
    engine.say("WAKE UP.")

    engine.runAndWait()
    engine.stop()
    
def crop_eye_dlib(img):
    IMG_SIZE = 50
    faces = 0
    roi = img.copy()

    try:
        image_array = img.copy()
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if(len(rects) < 0):
            return faces, 0
        for (i, rect) in enumerate(rects):
            faces = len(rects)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            coor_i, coor_j = (43, 48)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[coor_i:coor_j]]))
            roi = gray[y-(3*h):y+(2*h), x-w:x +(2*w)]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    except Exception as e:
            print(e)
    
    return faces, roi

def crop_mouth_yawn_dlib(img):
    IMG_SIZE = 50
    faces = 0
    roi = img.copy()

    try:
        image_array = img.copy()
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if(len(rects) < 0):
            return faces, 0
        for (i, rect) in enumerate(rects):
            faces = len(rects)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            coor_i, coor_j = (48, 68)
            (x, y, w, h) = cv2.boundingRect(np.array([shape[coor_i:coor_j]]))
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

    except Exception as e:
            print(e)
            
    return faces, roi

def predictEyes(roi):
    roi = np.array(roi).reshape(-1,50,50,1)
    roi = roi.astype('float32') / 255.0
    pred = model_eye.predict(roi)
    if pred < 0.5:
        return 0
    return 1

def predictYawn(roi):
    roi = np.array(roi).reshape(-1,50,50,1)
    roi = roi.astype('float32') / 255.0
    pred = model_yawn.predict(roi)
    if pred < 0.5:
        return 0
    return 1

def settings_page():
    global RECEIVER_EMAIL
    RECEIVER_EMAIL = simpledialog.askstring("Input", "Enter receiver address: ", initialvalue=RECEIVER_EMAIL, parent=root)
    if RECEIVER_EMAIL is None:
        RECEIVER_EMAIL = 'jcwong614@gmail.com'
    
def send_email(frame):
    
    if isinstance(frame, str):
        return
    
    msg = EmailMessage()
    msg['Subject'] = 'Driver Drowsiness Detected'
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.set_content('Instance of Driver Drowsiness Detected \n See image attached below...')
    
    success, encoded_image = cv2.imencode('.jpg', frame)
    image = encoded_image.tobytes()
    msg.add_attachment(image, maintype='image', subtype='jpg', filename='driver.jpg')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.send_message(msg)
        
    messagebox.showinfo("ALERT",  "Multiple Instances of Drowsiness Detected, sending email to your contact.")
    
def display_video():
    global th
    global th2
    global counter
    ret, frame = vid.read()
    text_eye = 'N/A'
    text_yawn = 'N/A'
    faces_eye, roi_eye = crop_eye_dlib(frame)
    faces_yawn, roi_yawn = crop_mouth_yawn_dlib(frame)

    #check if dlib detects any faces
    if (faces_eye <= 0 or faces_yawn <= 0):
        text_eye = 'N/A'
        text_yawn = 'N/A'
    else:
        pred_eye =  predictEyes(roi_eye)
        pred_yawn =  predictYawn(roi_yawn)

        if pred_eye == 0:
            text_eye = 'closed'
        else:
            text_eye = 'open'

        if pred_yawn == 0:
            text_yawn = 'no yawn'
        else:
            text_yawn = 'yawn'
            
        if pred_eye == 0 or pred_yawn == 1:
            counter += 1
            if counter > 3:
                if not th.is_alive() and not muted:
                    th = Thread(target=play_warning, daemon = True)
                    th.start()
                if not th2.is_alive():
                    th2 = Thread(target=send_email, args=[frame], daemon = True)
                    th2.start()
                
        else:
            counter = 0

    cv2.putText(frame, 'eye status = '+text_eye, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.putText(frame, 'yawn status = '+text_yawn, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert image to PhotoImage and apply on webcam_label 
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image = img)
    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)

    # Repeat after an interval to capture continiously
    webcam_label.after(10, display_video)

 # define a video capture object
vid = cv2.VideoCapture(0)
if not vid.isOpened():
    messagebox.showerror("Error","Unable to open webcam/camera.")
else:
    # declare variables
    font = ("bahnschrift",12)
    title_font = ("bahnschrift",20,"bold")
    muted = False
    RECEIVER_EMAIL = 'jcwong614@gmail.com'
    
    engine = pyttsx3.init()
    th = Thread(target=play_warning, args=[''],daemon = True)
    th2 = Thread(target=send_email, args=[''],daemon = True)
    # prepare root window
    root = Tk()
    root.title('Driver Drowsiness Detector')
    root.geometry("800x690")
    root.config(bg="slategrey")
    root.resizable(False, False)

    # prepare frames
    top_frame = Frame(root,bg="slategrey",pady=10)
    top_frame.pack(anchor=CENTER,side=TOP)

    main_frame = Frame(root,bg="slategrey")
    main_frame.pack(fill=X)

    buttom_frame = Frame(root,bg="slategrey",pady=20)
    buttom_frame.pack(anchor=CENTER,side=BOTTOM)

    ##### top frame #####

    title_label = Label(top_frame, font=title_font, text="Driver Drowsiness Detector", bg="slategrey", fg="white")
    title_label.pack()

    ##### main frame #####

    webcam_label = Label(main_frame)
    webcam_label.pack()

    #settings button
    settings_photo = PhotoImage(file=r"Shared\Icons\settings.png")
    btn_settings = Button(buttom_frame, font=font, image=settings_photo, padx=50, pady=10, cursor="hand2", bg="white", relief=FLAT, overrelief=SOLID, command=settings_page)
    btn_settings.grid(row=0,column=0,padx=20)

    
    # volumn button
    mute_photo = PhotoImage(file=r"Shared\Icons\mute.png")
    volume_photo = PhotoImage(file=r"Shared\Icons\unmute.png")
    btn_volume = Button(buttom_frame, image=volume_photo, relief=FLAT, overrelief=SOLID, cursor="hand2", command=mute_music)
    btn_volume.grid(row=0,column=1,padx=20)

    display_video()

    # place window at center when opening
    root.eval('tk::PlaceWindow . center')

    root.mainloop()
