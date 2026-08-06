import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.8, maxHands=2)

#capture local web cam, adjust webcam if needed
cam = 0 # primary cam for laptop 
vid = cv2.VideoCapture(cam)

while True:
    success, img = vid.read()
    img, bboxes = detector.findHands(img)

    if bboxes:
        center = bboxes[0]['center']

    if not success:
        break
    
    cv2.imshow("Video of falling tree", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    