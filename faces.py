import cv2
import numpy as np

img = cv2.imread('people.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('Net-work.xml')

results = faces.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), thickness=2)

cv2.imshow("faces", img)
cv2.waitKey(0)

