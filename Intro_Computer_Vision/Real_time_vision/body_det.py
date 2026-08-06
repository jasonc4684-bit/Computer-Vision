import cv2
from cvzone.PoseModule import PoseDector

detector = PoseDector()

#capture local web cam, adjust webcam if needed
cam = 0 # primary cam for laptop 
vid = cv2.VideoCapture(cam)

while True:
    success, img = vid.read()
    img = detector.findPose(img)
    lmlist, bboxes = detector.findPosition()

    if hands:
        hand1 = hands[0]
        #landmark list on hand
        lmlist = hand1['lmList']
        handType = hand1['type']

    if not success:
        break
    
    cv2.imshow("hand det", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    