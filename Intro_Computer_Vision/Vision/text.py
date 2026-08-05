import cv2

img = cv2.imread("../Computer_vision_essential/intro/school.jpg")

cv2.putText(img, "This is one high school at Virginia", 
            (10,20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,200), 2)

cv2.imshow("img", img)

cv2.waitKey(0)
