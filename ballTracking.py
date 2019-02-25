from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("-v", "--video", help="path to video file")
args = vars(ap.parse_args())

orangeLower = (0,66,147)
orangeUpper = (77,255,255)

yellowLower = (22,73,62)
yellowUpper = (55,255,255)

bellPepperLow = (0,124,110)
bellPepperHigh = (16,255,255)

redCupLow = (0,114,162)
redCupHigh = (26,255,255)

pts = deque([(-99,-99)] * args["buffer"], maxlen=args["buffer"])
counter = 0
dX = 0
actX = None
actY = None
coords = None
dXo = None

if not args.get("video", False):
    vs = VideoStream(src=0).start()
 
else:
    vs = cv2.VideoCapture(args["video"])
 
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if radius > 8:
            cv2.circle(frame, center, 5, (255,0,255), -1)
            pts.appendleft(center)

    for i in np.arange(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        if counter >= 20 and i == 1 and pts[i] != (-99,-99):
            dX = pts[-10][0] - pts[i][0]
            dXo  = pts[-11][0] - pts[-20][0]
            actX = pts[-10][0]
            actY = pts[-10][1]
            
        cv2.line(frame, pts[i-1], pts[i], (195,0,255), 1)
        
    if dXo is not None and (np.sign(dX) != np.sign(dXo)):
        coords = (actX,actY)
    elif coords != None:
        if coords != (-99,-99):
            cv2.circle(frame,coords, 2, (65,255,255), 7)
            print (coords)
            coords = None
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
    
    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()