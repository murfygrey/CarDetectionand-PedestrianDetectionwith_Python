import cv2
import os
import numpy as np
from PIL import Image
import pickle

#Addd image
#img_file ='car2.jpg'
video = cv2.VideoCapture('tesla.mp4')
#video = cv2.VideoCapture('Pedestrians.mp4')

#our trained car classifier
car_tarcker_file = 'Cartracting.xml'
pedestrian_tracker_file ='hogcascade_pedestrians.xml'

#our car classifier
car_tracker = cv2.CascadeClassifier(car_tarcker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


 #run foreverv until crash or car stops
while True:

    #Read current frame
    read_successful,frame = video.read()
    #safe coding
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #Detect Cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around cars and pedestrians
    for (x , y ,h , w) in cars:
       cv2.rectangle(frame, (x, y), (x+w , h+y), (0, 0, 255), 2 )
    for (x , y ,h , w) in pedestrians:
       cv2.rectangle(frame, (x, y), (x+w , h+y), (0, 255, 255), 2 )

    #Detect the car
    key = cv2.imshow('Car Detector ' ,frame)

    #adding waitkey
    cv2.waitKey(1)  
    #Quit
    if key==81 or key==113 :
       break
# release video capture
video.release()
 
   #create opencv image
   #img = cv2.imread(img_file)

   #image to black and white
    #black_n_white = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 





#print(cars)



print("Code Complted")