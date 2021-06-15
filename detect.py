import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras.models import load_model

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 1)
    img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=1, thresholdType=1, blockSize=11, C=2)
    return img

def find_countour(img, pp):
    img_copy = img.copy()
    contours = cv2.findContours(pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = cv2.drawContours(img_copy, contours[1], -1, (0, 255, 0), 3)
    return img_contour, contours

def bigContour(contours):
    big = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50: #if too small, will find noise
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            if area > max_area and len(approx)==4: #checking of rect/square
                big = approx
                max_area = area
    return big, max_area

# order of the 4 points should be the same always. reorder() ensures this
def reorder(points):
    points = points.reshape((4,2))
    points_new = np.zeros((4,1,2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] =  points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def function(img, contours):
    img_big_contour = img.copy()
    big, max_area = bigContour(contours[1])
    if big.size!=0:
        big = reorder(big)
        img_big_contour = cv2.drawContours(img_big_contour, big, -1, (255,255,0), 20) #draw the biggest contour
        # plt.title('img_big_contour')
        # plt.imshow(img_big_contour, cmap='gray')
        # plt.show()
        # preapres points for warp
        pts1 = np.float32(big) 
        pts2 = np.float32([[0,0], [252,0], [0,252], [252, 252]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_img = cv2.warpPerspective(img, matrix, (252, 252))
        # plt.title('warped_1')
        # plt.imshow(warped_img, cmap='gray')
        # plt.show()
        warped_img = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        # plt.title('warped_2')
        # plt.imshow(warped_img, cmap='gray')
        # plt.show()
        return warped_img

def postprocess(image):
    for i,img in enumerate(image):
        mean = np.mean(img)
        std = np.std(img)
        image[i] = (img-mean)/std
    return image

def split_boxes(image):
    rows = np.vsplit(image, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            
            boxes.append(box)
    return boxes

def getbox(img):
    boxes = split_boxes(img)
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            boxes[i][j][0]=0
            boxes[i][j][-1]=0      

    nums = np.array([1 if np.sum(boxes[i].flatten())/(255*81)>0.7 else 0 for i in range(len(boxes))])

    return np.array(boxes), nums

def predict(boxes):
    model = load_model('model.h5')
    predicted_numbers = []
    predicted_proba = []
    for b in boxes:
        b = np.reshape(b,(1,28,28))
        predicted_numbers.append(model.predict_classes(b))
        predicted_proba.append(model.predict_proba(b))
    return predicted_numbers,predicted_proba