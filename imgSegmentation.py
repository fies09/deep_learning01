#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:20
# @Author     : fany
# @Project    : PyCharm
# @File       : imgSegmentation.py
# @description: 使用OpenCV进行图像分割
import cv2

# 读取图像
image = cv2.imread('input_image.jpg')

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值分割
_, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# 显示原始图像和分割后的图像
cv2.imshow('Original Image', gray_image)
cv2.imshow('Segmented Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
