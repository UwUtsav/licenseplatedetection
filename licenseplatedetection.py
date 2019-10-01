from PIL import Image
import pytesseract
import argparse
import cv2
import os
from copy import deepcopy
import numpy as np
import  imutils
import sys
import pandas as pd
import time

image = cv2.imread('E:\python\car5.jpg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()#4

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break



mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)


config = ('-l eng --oem 1 --psm 3')



text = pytesseract.image_to_string(new_image, config=config)

raw_data = {'date':[time.asctime( time.localtime(time.time()))],'':[text]}
#raw_data = [time.asctime( time.localtime(time.time()))],[text]

df = pd.DataFrame(raw_data)
df.to_csv('E:\python\data.csv',mode='a')


print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
