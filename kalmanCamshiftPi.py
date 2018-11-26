import cv2
import argparse
import sys
import math
import numpy as np
import RPi.GPIO as GPIO
import time


#####################################################################

keep_processing = True
selection_in_progress = False
pause = False
font = cv2.FONT_HERSHEY_SIMPLEX
# FONT_HERSHEY_SIMPLEX = 0,
# FONT_HERSHEY_PLAIN = 1,
# FONT_HERSHEY_DUPLEX = 2,
# FONT_HERSHEY_COMPLEX = 3,
# FONT_HERSHEY_TRIPLEX = 4,
# FONT_HERSHEY_COMPLEX_SMALL = 5,
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
# FONT_HERSHEY_SCRIPT_COMPLEX = 7, # support interactive region selection

# select a region using the mouse
boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)

def calculateAngle(angle):
    calc = float(angle) / 10.0 + 2.5#((angle/180.0) + 1.0) * 5.0
    return calc

def on_mouse(event, x, y, flags, params):

    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []
        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)

def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(x), np.float32(y)], np.float32)

def centerJ(x, y, x1, y1):
    centerX = int(np.floor((x+x1)/2))
    centerY = int(np.floor((y+y1)/2))
    return(centerX, centerY)

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass


#set up servos
GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)

pan = GPIO.PWM(12, 50)
panNum = 50
panAngle = calculateAngle(panNum)
pan.start(panAngle)
           
tilt = GPIO.PWM(11, 50)
tiltNum = 40
tiltAngle = calculateAngle(tiltNum)
tilt.start(tiltAngle)
time.sleep(1)


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

windowName = "Kalman Object Tracking" # window name
windowNameSelection = "initial selected region"

# init kalman filter object

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03

measurement = np.array((2,1), np.float32)
prediction = np.zeros((2,1), np.float32)

print("\nObservation in image: BLUE")
print("Prediction from Kalman: RED\n")

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

# if (((args.video_file) and (cap.open(str(args.video_file))))
#     or (cap.open(args.camera_to_use))):
if True:

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL)

    # set sliders for HSV selection thresholds
    s_lower = 60
    s_upper = 255
    v_lower = 32
    v_upper = 255

    # set a mouse callback

    cv2.setMouseCallback(windowName, on_mouse, 0)
    cropped = False

    # Setup the termination criteria for search, either 10 iteration or
    # move by at least 1 pixel pos. difference
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while (keep_processing):

        if pause == False:
        # if video file successfully open then read frame from video

            if (cap.isOpened):
                ret, frame = cap.read()

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount()

            # select region using the mouse and display it

            if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):
                crop = frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()

                h, w, c = crop.shape   # size of template
                if (h > 0) and (w > 0):
                    cropped = True

                    # convert region to HSV

                    hsv_crop =  cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                    # select all Hue (0-> 180) and Sat. values but eliminate values with very low
                    # saturation or value (due to lack of useful colour information)

                    mask = cv2.inRange(hsv_crop, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))))
                    # mask = cv2.inRange(hsv_crop, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

                    # construct a histogram of hue and saturation values and normalize it

                    crop_hist = cv2.calcHist([hsv_crop],[0, 1],mask,[180, 255],[0,180, 0, 255])
                    cv2.normalize(crop_hist,crop_hist,0,255,cv2.NORM_MINMAX)

                    # set intial position of object

                    track_window = (boxes[0][0],boxes[0][1],boxes[1][0] - boxes[0][0],boxes[1][1] - boxes[0][1])

                    cv2.imshow(windowNameSelection,crop)

                # reset list of boxes

                boxes = []

            # interactive display of selection box

            if (selection_in_progress):
                top_left = (boxes[0][0], boxes[0][1])
                bottom_right = (current_mouse_position[0], current_mouse_position[1])
                cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2)

            # if we have a selected region

            if (cropped):

                # convert incoming image to HSV

                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # back projection of histogram based on Hue and Saturation only

                img_bproject = cv2.calcBackProject([img_hsv],[0,1],crop_hist,[0,180,0,255],1)
                #cv2.imshow(windowName2,img_bproject)

                # apply camshift to predict new location (observation)
                # basic HSV histogram comparision with adaptive window size
                # see : http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html
                ret, track_window = cv2.CamShift(img_bproject, track_window, term_crit)

                # draw observation on image - in BLUE
                x,y,w,h = track_window
                frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
                cv2.rectangle(frame,(10,0),(90,90),(255,255,255), -1)
                # extract centre of this observation as points

                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                # (cx, cy), radius = cv2.minEnclosingCircle(pts)
                centerXY = center(pts)
                cv2.putText(frame, 'Mx: ' + str(int(np.floor(centerXY[0]))), (20, 20), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, 'My: ' + str(int(np.floor(centerXY[1]))), (20, 40), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                
                
                if centerXY[1] > 350: #go down
                    tiltNum = tiltNum + 1
                    tiltAngle = calculateAngle(tiltNum)
                    tilt.ChangeDutyCycle(tiltAngle)

                if centerXY[1] < 150: #go down
                    tiltNum = tiltNum - 1
                    tiltAngle = calculateAngle(tiltNum)
                    tilt.ChangeDutyCycle(tiltAngle)

                if centerXY[0] > 450: #go right
                    panNum = panNum - 1
                    panAngle = calculateAngle(panNum)
                    pan.ChangeDutyCycle(panAngle)
                if centerXY[0] < 200: #go left
                    panNum = panNum + 1
                    panAngle = calculateAngle(panNum)
                    pan.ChangeDutyCycle(panAngle)
                
                
                # use to correct kalman filter
                kalman.correct(centerXY)

                # get new kalman filter prediction

                prediction = kalman.predict()

                # draw predicton on image
                frame = cv2.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (0,0,255),2)
                centerXYJ = centerJ((prediction[0]-(0.5*w)),(prediction[1]-(0.5*h)), (prediction[0]+(0.5*w)), (prediction[1]+(0.5*h))) 
                cv2.putText(frame, 'Px: ' + str(centerXYJ[0]), (20, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Py: ' + str(centerXYJ[1]), (20, 80), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                #out.write(frame)

            else:

                # before we have cropped anything show the mask we are using
                # for the S and V components of the HSV image

                img_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # select all Hue values (0-> 180) but eliminate values with very low
                # saturation or value (due to lack of useful colour information)

                mask = cv2.inRange(img_hsv, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))))

                #cv2.imshow(windowName2,mask)

            # display image
            cv2.imshow(windowName,frame)

        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF
        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif (key == ord('p')):
            if pause == True:
                pause = False
            else:
                pause= True
    # close all windows
    #out.release()
    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

#####################################################################