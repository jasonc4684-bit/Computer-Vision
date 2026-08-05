import cv2
import cvzone
import numpy as np
from cvzone.Utils import findContours

img = cv2.imread("Intro_Computer_Vision/Vision/img.jpg")

imgBlur = cv2.GaussianBlur(img, (75, 75), 0)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurCanny = cv2.Canny(imgBlur, 125, 124)
grayCanny = cv2.Canny(imgGray, 150, 200)

blur_dilate = cv2.dilate(blurCanny, np.ones((6, 6), np.uint8), iterations=1)
gray_dilate = cv2.dilate(grayCanny, np.ones((6, 6), np.uint8), iterations=1)

gray_imgContours, conFound = findContours(img, gray_dilate, filter= None, drawCon= True)
blur_imgContours, conFound = findContours(img, blur_dilate, filter= None, drawCon= True)

imgStack = cvzone.stackImages([img, imgGray, grayCanny, gray_dilate, gray_imgContours, 
                                img, imgBlur, blurCanny, blur_dilate, blur_imgContours], 5, 0.25)

cv2.imshow("img", imgStack)
cv2.waitKey(0)