import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl

img = cv2.imread('550xNx1623477926.jpeg.pagespeed.ic.jXnRJ9ZceF.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

img_filters = cv2.bilateralFilter(gray, 11, 15, 15)
edges = cv2.Canny(img_filters, 30, 200)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx_number = cv2.approxPolyDP(c, 10, True)
    if len(approx_number) == 8:
        pos = approx_number
        break

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, (255, 0, 0), -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
crop = gray[x1:x2, y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(crop)

res = text[0][-2]
lebel = cv2.putText(img, res, (x1 - 200, y2 + 160), cv2.FONT_HERSHEY_TRIPLEX,  3, (0,255,0), 1)
lebel = cv2.rectangle(img,(x1,x2), (y1,y2), (255,255,0), 1)

pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB))
pl.show()
