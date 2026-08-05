import cv2
import cvzone
import numpy as np
from cvzone.Utils import findContours

img = cv2.imread("Intro_Computer_Vision/Vision/img.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grayCanny = cv2.Canny(imgGray, 100, 250)

gray_dilate = cv2.dilate(grayCanny, np.ones((5, 5), np.uint8), iterations=2)

gray_imgContours, conFound = findContours(img, gray_dilate, filter= None, drawCon= True)

imgPresentSize = 0.6 # for most laptop, 
Columns = 3
imgStack = cvzone.stackImages([img, imgGray, grayCanny, gray_dilate, gray_imgContours],
                                                        Columns, imgPresentSize)

print(conFound)
cv2.imshow("img", imgStack)
cv2.waitKey(0)