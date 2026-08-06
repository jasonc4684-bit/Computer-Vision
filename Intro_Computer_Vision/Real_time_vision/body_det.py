import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

#capture local web cam, adjust webcam if needed
cam = 0 # primary cam for laptop 
vid = cv2.VideoCapture(cam)

while True:
    success, img = vid.read()
    img = detector.findPose(img)
    lmlist, bboxes = detector.findPosition(img)

    if bboxes:
        center = bboxes['center']
    if not success:
        break
    
    cv2.imshow("body det", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    