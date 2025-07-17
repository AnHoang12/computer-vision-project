import cv2 

img = cv2.imread("/home/anhoang/Basic_DL/cv/opencv/basic/apple.jpg")

rotate = cv2.rotate(img,cv2.ROTATE_180)

cv2.imshow('img',rotate)
cv2.waitKey(0)