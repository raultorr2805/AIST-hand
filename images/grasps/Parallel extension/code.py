# OpenCV code
import cv2
img = cv2.imread('Cracker-box-right-subj2-1.jpg')
# HXRI code
from hxri import HandDetector
h = HandDetector.MediapipeHand();
landmarks, vectors, angles = h.getHandsInfo(img)
print (landmarks)
# Display image with drawing hands
cv2.imshow("Result", h.drawLandmarks(img))
cv2.waitKey(0)
