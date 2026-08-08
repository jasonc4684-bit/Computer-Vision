import cv2

img = cv2.imread("school.jpg")
cv2.imshow("Image school", img)

#waiting
cv2.waitKey(0)
