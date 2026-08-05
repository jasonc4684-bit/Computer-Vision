import cv2

img = cv2.imread("../Computer_vision_essential/intro/school.jpg")

#         [y1:y2, x1:x2]
crop = img[100:2000, 10:1200]

cv2.imshow("img", img)
cv2.imshow("crop img", crop)
cv2.waitKey(0)
