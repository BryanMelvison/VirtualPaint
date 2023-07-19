#Importing necessary libraries
import cv2
import mediapipe as mp
# import numpy as np

#Callback function for the trackbar
def empty(a):
    pass

#Draw the painting on the Canvas
def draw(paintings, img):
    for point in paintings:
        cv2.circle(img, (point[0], point[1]), 15, point[2], cv2.FILLED)

#Initializing the webcam dimensions
cap = cv2.VideoCapture(0)
cap.set(3,880)
cap.set(4,780)
cap.set(10,100)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#Creating the color picket window for RGB colors along with an eraser
cv2.namedWindow("Color Picker")
cv2.resizeWindow("Color Picker",400, 190)
cv2.createTrackbar("Blue","Color Picker" ,0 ,255 ,empty)
cv2.createTrackbar("Green","Color Picker" ,0 ,255 ,empty)
cv2.createTrackbar("Red","Color Picker" ,0 ,255 ,empty)
cv2.createTrackbar("Clear","Color Picker" ,0 ,1 ,empty)

#To record the difference from a frame to another and the points 
static = None
paintings = []

while True:
    ret, img = cap.read()

    #get tracker position denoting the colors picked
    blue_val = cv2.getTrackbarPos("Blue","Color Picker")
    green_val = cv2.getTrackbarPos("Green","Color Picker")
    red_val = cv2.getTrackbarPos("Red","Color Picker")
    clear = cv2.getTrackbarPos("Clear", "Color Picker")

    #Color shown on the video
    cv2.putText(img,"Color:",(20,30), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    cv2.circle(img,(90,20), 10, (blue_val,green_val,red_val), cv2.FILLED)
    
    #Convert color image to a gray then to a gaussian blur image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if static is None:
        static = gray
        continue
    
    #Difference between static background and current frame(gaussian blur)
    diff = cv2.absdiff(static, gray)
    thresh = cv2.threshold(diff,10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations = 2)

    #Find contours(Detect motions)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Draw out the motion
    for cnts in contours:
        M = cv2.moments(cnts)

        if cv2.contourArea(cnts) < 4000:
            continue

        cv2.drawContours(img, cnts, -1, (0,255,0), 1)

        # cv2.circle(img, cnts, 10, (blue_val,green_val,red_val),3)
        x,y,w,h = cv2.boundingRect(cnts)

        #find center point of the contour
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cv2.circle(img, (cx, y), 15, (blue_val,green_val,red_val), cv2.FILLED)

            #Paint the parts touched by the center of the contour
            paintings.append([cx,y, (blue_val,green_val,red_val)])
        
        cv2.rectangle(img,(x,y), (x+w, y + h), (0, 255, 0),1)

    #If eraser is True:
    if clear == 1:
        paintings = []

    #gui side
    if len(paintings) != 0:
        draw(paintings,img)
    
    static = gray

    #To play around with, hence commented
    # cv2.imshow("D", diff)
    # cv2.imshow("T", thresh)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()