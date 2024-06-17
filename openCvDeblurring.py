#!/usr/bin/env python
# -*- coding = utf-8 -*-
# @Time       : 2023/10/13 15:19
# @Author     : fany
# @Project    : PyCharm
# @File       : openCvDeblurring.py
# @description: openCvDeblurring
from skimage import io, color, restoration
import matplotlib.pyplot as plt

# 读取图像
image = io.imread('input_image.jpg')

# 将图像转换为灰度图
gray_image = color.rgb2gray(image)

# 使用Richardson-Lucy去模糊算法进行去模糊
deconvolved_image, _ = restoration.richardson_lucy(gray_image, psf)

# 显示原始图像和去模糊后的图像
plt.figure()
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.figure()
plt.imshow(deconvolved_image, cmap='gray')
plt.title('Deblurred Image')
plt.show()
