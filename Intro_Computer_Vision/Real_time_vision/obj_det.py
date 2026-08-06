from ultralytics import YOLO
import cv2
import cvzone

#capture local web cam, adjust webcam if needed
cam = 0
vid = cv2.VideoCapture(cam)

while True:
    success, img = vid.read()

    if not success:
        break
    cv2.imshow("obj_det", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break