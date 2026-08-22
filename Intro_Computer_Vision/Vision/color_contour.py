import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
from cvzone.Utils import findContours

Cam = 0
vid = cv2.VideoCapture(Cam)

#                                    True to reset color det.
myColorFinder = ColorFinder(trackBar=True)

# empty the bracket to reset color det.
hsvVal = {}

while True:
    success, img = vid.read()

    if not success:
        break

    #updating the HSV
    imgColor, mask = myColorFinder.update(img, hsvVal)

    #added features for detecting pos and draw
    img_contour, conFound = findContours(img, mask)

    #display multi imgs.
    imgStack = cvzone.stackImages([img, imgColor, mask, img_contour], 4, 0.2)

    #given x and y position 
    if conFound:
        print(conFound[0]['center'])

    cv2.imshow("Color detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    