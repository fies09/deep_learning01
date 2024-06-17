#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:18
# @Author     : fany
# @Project    : PyCharm
# @File       : openCvDenoising.py
# @description: openCvDenoising
import cv2
import numpy as np

# 读取图像
image = cv2.imread('input_image.jpg', cv2.IMREAD_COLOR)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯滤波进行去噪
denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 显示原始图像和去噪后的图像
cv2.imshow('Original Image', gray_image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
