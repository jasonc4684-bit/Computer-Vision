import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

Cam = 0
vid = cv2.VideoCapture(Cam)

# a debug-like in color and mask
myColorFinder = ColorFinder(trackBar=False)

hsvVal = {'hmin': 108, 'smin': 67, 'vmin': 63, 'hmax': 179, 'smax': 255, 'vmax': 255}

while True:
    success, img = vid.read()

    if not success:
        break

    imgColor, mask = myColorFinder.update(img, hsvVal)

    columns = 3
    displaySize = 0.3
    imgStack = cvzone.stackImages([img, imgColor, mask], columns, displaySize)

    cv2.imshow("Color detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
