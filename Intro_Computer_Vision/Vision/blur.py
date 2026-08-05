import cv2

img = cv2.imread("Intro_Computer_Vision/Intro/school.jpg")
Blur = cv2.GaussianBlur(img, (15, 15), 0)
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)
cv2.imshow("Blur img", Blur)
cv2.waitKey(0)
