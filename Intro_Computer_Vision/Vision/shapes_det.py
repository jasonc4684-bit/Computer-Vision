import cv2
import cvzone
import numpy as np
from cvzone.Utils import findContours

Img = "Intro_Computer_Vision/Vision/img.jpg"
img = cv2.imread(Img)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lower_bound = 100
high_bound = 250
grayCanny = cv2.Canny(imgGray, lower_bound, high_bound)

newArraySize = (5,5)
gray_dilate = cv2.dilate(grayCanny, np.ones(newArraySize, np.uint8), iterations=2)

gray_imgContours, conFound = findContours(img, gray_dilate, filter= None, drawCon= True)

imgPresentSize = 0.6 # for most laptop, 
Columns = 3
imgStack = cvzone.stackImages([img, imgGray, grayCanny, gray_dilate, gray_imgContours],
                                                        Columns, imgPresentSize)

print(conFound)
cv2.imshow("img", imgStack)
cv2.waitKey(0)
