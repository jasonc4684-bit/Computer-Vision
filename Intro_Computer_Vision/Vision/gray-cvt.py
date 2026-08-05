import cv2

img = cv2.imread("Intro_Computer_vision/intro/school.jpg")
Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)
cv2.imshow("Gray img", Gray)
cv2.waitKey(0)
