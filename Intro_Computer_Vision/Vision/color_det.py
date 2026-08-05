import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

vid = cv2.VideoCapture(0)

myColorFinder = ColorFinder(trackBar=True)

hsvVal = {}

while True:
    success, img = vid.read()

    if not success:
        break
    cv2.imshow("Color detection", img)

    imgColor, mask = myColorFinder.update(img, hsvVal)

    imgStack = cvzone.stackImages(img, imgColor, mask, 3, 0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    