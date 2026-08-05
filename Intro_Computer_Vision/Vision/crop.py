import cv2

#read files
IMG = "Intro_Computer_Vision/Intro/school.jpg"
img = cv2.imread(IMG)

#         [y1:y2, x1:x2]
y1 = 0
y2 = 10
x1=10
x2=20
crop = img[y1:y2, x1:x2]

cv2.imshow("img", img)
cv2.imshow("crop img", crop)
cv2.waitKey(0)
