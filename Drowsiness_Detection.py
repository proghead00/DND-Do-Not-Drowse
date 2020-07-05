import time
from pygame import mixer
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
thresh = 0.25
frame_check = 40
#initializing dlib's face detector (HOG-based) and then creating the facial landmark predictor.
detect = dlib.get_frontal_face_detector() #it returns the default face detector.
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #it takes in an image region containing some object and outputs a set of point locations that define the pose of the object.
mixer.init()
#extracting the left and right eye's (x, y)-coordinates.
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
#VideoCapture:Class for video capturing from video files, image sequences or cameras.
#cv2.VideoCapture(device):id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.
flag=0 # initializing the frame counter.
while True:
    ret, frame=cap.read() #.read():Grabs, decodes and returns the next video frame.
    frame = imutils.resize(frame, width=450) #resizing image to maximum of 450 pixels. cv2.resize() can also do the same but in order to
                                             #maintain aspect ratio imutils.resize() is used.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #.cvtColor():Converts BGR image into GRAY image.
    subjects = detect(gray, 0) #detects faces in the grayscale frame.
    #looping over the face detections.
    for subject in subjects:
        shape = predict(gray, subject) #determining the facial landmarks for the face region.
        shape = face_utils.shape_to_np(shape) #converting the facial landmark (x, y)-coordinates to a NumPy array.
        #extracting the left and right eye coordinates.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        #using the coordinates for calculating aspect ratio for both eyes.
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        #calculating average aspect ratio.
        ear = (leftEAR + rightEAR) / 2.0
        #computing the convex hull for the left and right eye(convex hull:-the smallest convex shape enclosing a given shape).
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #visualizing each of the eyes.
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) #.drawContours():-Draws contours outlines or filled contours.
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) #(thickness==negative):-the contour interiors are drawn.(maxLevel==1):-the 
                                                                    #function draws the contour(s) and all the nested contours.
        if ear < thresh:
            flag += 1 #incrementing the blink frame counter.
            if flag >= frame_check:
                mixer.music.load("beep2.mp3")
                time.sleep(.01)
                mixer.music.play()
                
                #cv2.putText(image,text,org,fontFace,fontScale,color,thickness)
                cv2.putText(frame, "", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2) #to draw a text string.
                cv2.putText(frame, "       WAKE UP    ", (10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag=0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        #to cleanup the camera and close any open windows
        cv2.destroyAllWindows()
        cap.release()
        break
