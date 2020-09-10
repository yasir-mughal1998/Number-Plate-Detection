import cv2
import numpy as np

numPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture("gettyimages-807742384-640_adpp.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)

if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    numPlate = numPlateCascade.detectMultiScale(frame,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in numPlate:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, 'Number Plate', (x - 10, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (20, 20))

    if ret == True:
        cv2.imshow('Video',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
