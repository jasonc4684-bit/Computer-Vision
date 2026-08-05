import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
from cvzone.Utils import findContours

vid = cv2.VideoCapture(0)

myColorFinder = ColorFinder(trackBar=False)

hsvVal = {'hmin': 108, 'smin': 67, 'vmin': 63, 'hmax': 179, 'smax': 255, 'vmax': 255}

while True:
    success, img = vid.read()

    if not success:
        break

    imgColor, mask = myColorFinder.update(img, hsvVal)
    img_contour, conFound = findContours(img, mask)

    imgStack = cvzone.stackImages([img, imgColor, mask, img_contour], 4, 0.2)

    #given x and y position 
    if conFound:
        print(conFound[0]['center'])

    cv2.imshow("Color detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    