import cv2 as cv
img = cv.imread("img/127522431331980736517_00-006-153.png")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window