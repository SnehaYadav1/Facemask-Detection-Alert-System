#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary linbraries
from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox   # to create warning pop up window
import smtplib       # used to define an smtp client session object that can be used to send an alert email to concerned authorities, if the person detected is not wearing th mask 

# Initialize Tkinter
root = tkinter.Tk()  # initializing the tkinter in order to create a tk root widget which is a window with a little bar and other decoration provided by the window manager, the root wifget has to be created before any other widget and there can only be one root widget   
root.withdraw()    # then we are hiding the root window from the screen without destroying it using withdraw function (if we don't use this withdraw function, then it will cause the app to create an empty root window always open , which we don't want)

#Load trained deep learning model
model = load_model('face_mask_detection_using_cnn.h5')

#Classifier to detect face
face_det_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   # using open cv's haarcascade front face classifier to detect the face in the video frame so that first we locate the region of interest and then we can use our deep learning model loaded up , to predict if the prson is wearing mask or not 

# Capture Video
vid_source=cv2.VideoCapture(0)  # return video from the first webcam of your computer

# Dictionaries containing details of Wearing Mask and Color of rectangle around face. If wearing mask then color would be 
# green and if not wearing mask then color of rectangle around face would be red
text_dict={0:'Mask ON',1:'No Mask'}   # 0 an 1 are keys
rect_color_dict={0:(0,255,0),1:(0,0,255)}

SUBJECT = "Subject"   
TEXT = "One Visitor violated Face Mask Policy. See in the camera to recognize user.A Person has been detected without a face mask in the Hotel Lobby Area 6. Please Alert the authorities."


# While Loop to continuously detect camera feed
while(True):

    ret, img = vid_source.read()  # here ret is a boolean value that tells whether any frame is returned from the video feed  and frame is a variable that stores a returned frame. (You will get none value if there is no frame returned )    
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # converting color image feed into grayscale using cvtColor function 
    faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)   # to get the region of interest in the frame ,  1.3(resucing the image by 30% each time it is scaled ) is scale factor which depicts parameter specifying how much the image size is reduced at each image scale, 5 is min neighbours that depicts how many neighbours each candidate rectangle should have to retain( i.e. if we have multiple faces in the same region then it may draw multiple rectangles , so through this parameter we are telling that it should consider it as one face)   

    for (x,y,w,h) in faces: # coordinates of the face

        face_img = grayscale_img[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)

        label=np.argmax(result,axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img, text_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)  # used to put text as mask on and no mask on the rectabgle
    
        # If label = 1 then it means wearing No Mask and 0 means wearing Mask
        if (label == 1):
            # Throw a Warning Message to tell user to wear a mask if not wearing one. This will stay
            # open and No Access will be given He/She wears the mask
            messagebox.showwarning("Warning","Access Denied. Please wear a Face Mask")

            # Send an email to the administrator if access denied/user not wearing face mask 
            message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('facemask167@gmail.com','Face@123')
            mail.sendmail('facemask167@gmail.com','facemask167@gmail.com',message)
            mail.close
        else:
            pass
            break

    cv2.imshow('LIVE Video Feed',img)
    key=cv2.waitKey(1)

    if(key==27):    # 27 is escape key here
        break

cv2.destroyAllWindows()
source.release()


# In[ ]:




