import numpy as np
# import pandas as pd
import cv2
# import matplotlib.pyplot as plt
# import os
from tensorflow.keras.models import load_model

class SudokuProcess:
    def __init__(self, sudoku_img_path):
        self.img = cv2.imread(sudoku_img_path)
        self.pp = None
        self.contours = None
        self.img_contour = None
        self.warped_img = None

    def preprocess(self):
        self.img = cv2.resize(self.img,(282,282))
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5,5), 1)
        img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=1, thresholdType=1, blockSize=11, C=2)
        self.pp = img

    def find_countour(self):
        img_copy = self.img.copy()
        contours = cv2.findContours(self.pp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = cv2.drawContours(img_copy, contours[0], -1, (0, 255, 0), 3)
        self.img_contour, self.contours = img_contour, contours

    def bigContour(self):
        big = np.array([])
        max_area = 0
        for i in self.contours[0]:
            area = cv2.contourArea(i)
            if area > 50: #if too small, will find noise
                perimeter = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
                if area > max_area and len(approx)==4: #checking of rect/square
                    big = approx
                    max_area = area
        return big, max_area

    # order of the 4 points should be the same always. reorder() ensures this
    def reorder(self, points):
        points = points.reshape((4,2))
        points_new = np.zeros((4,1,2), dtype=np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] =  points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new

    def function(self):
        img_big_contour = self.img.copy()
        big, max_area = self.bigContour()
        if big.size!=0:
            big = self.reorder(big)
            img_big_contour = cv2.drawContours(img_big_contour, big, -1, (255,255,0), 20) #draw the biggest contour
            # plt.title('img_big_contour')
            # plt.imshow(img_big_contour, cmap='gray')
            # plt.show()
            # preapres points for warp
            pts1 = np.float32(big) 
            pts2 = np.float32([[0,0], [252,0], [0,252], [252, 252]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.warped_img = cv2.warpPerspective(self.img, matrix, (252, 252))
            # plt.title('warped_1')
            # plt.imshow(warped_img, cmap='gray')
            # plt.show()
            self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_RGB2GRAY)
            # plt.title('warped_2')
            # plt.imshow(warped_img, cmap='gray')
            # plt.show()

    def postprocess(self):
        for i,img in enumerate(self.warped_img):
            mean = np.mean(img)
            std = np.std(img)
            self.warped_img[i] = (img-mean)/std
        cv2.imshow('postproc', self.warped_img)
        cv2.waitKey(0)

    def split_boxes(self, image):
        rows = np.vsplit(image, 9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                
                boxes.append(box)
        return boxes

    def getbox(self):
        boxes = self.split_boxes(self.warped_img)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j][0]=0
                boxes[i][j][-1]=0      

        nums = np.array([1 if np.sum(boxes[i].flatten())/(255*81)>0.7 else 0 for i in range(len(boxes))])

        return np.array(boxes), nums

    def predict(self, boxes):
        model = load_model('model.h5')
        boxes = np.reshape(boxes, (-1,28,28,1))
        predicted_numbers = model.predict_classes(boxes)
        return predicted_numbers


