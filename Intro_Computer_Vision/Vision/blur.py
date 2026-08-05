import cv2

IMG = "Intro_Computer_Vision/Intro/school.jpg"

img = cv2.imread(IMG)

#higher () = more blury
Blur = cv2.GaussianBlur(img, (15, 15), 0)

#to gray
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)
cv2.imshow("Blur img", Blur)
cv2.waitKey(0)
